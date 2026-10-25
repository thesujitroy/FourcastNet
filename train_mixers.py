#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
# Derived from train.py (see LICENSE for the full license text and author list).

"""Mixer-comparison copy of train.py: FourCastNet with the token mixer chosen by config.

    python train_mixers.py --yaml_config config/mixer_comparison.yaml --config tucker_spectral \
        --run_num 0 --enable_amp [--override max_epochs=1 max_iters_per_epoch=20]

Same as train.py: data loader, LpLoss (relative L2) objective, Adam, per-epoch
cosine schedule, DDP, checkpoint/resume layout (exp_dir/<config>/<run_num>/), the
two-step fine-tuning path.

Different, for a controlled comparison (docs/superpowers/specs/2026-10-08-fourcastnet-
mixer-comparison-design.md):
  * model: networks/mixernet.py::MixerNet (== AFNONet for `mixer: {type: afno}`)
  * `seed` fixes the initial weights and the training-sample order (train.py seeds
    nothing, and does not shuffle at all on one GPU)
  * no apex / ruamel; `amp_dtype` picks bf16 (default) or fp16 under --enable_amp
  * validation on a fixed, evenly spaced subset of `n_valid_samples` over the test
    years (train.py: its first 20 batches, i.e. early January), with per-channel
    latitude-weighted RMSE (physical units) and ACC against time_means
  * HelioFM-style records next to the checkpoints: model.jsonl, epochs.jsonl,
    validation.jsonl; W&B names/groups from the config
  * resume keeps the best validation loss and cumulative GPU hours; checkpoints are
    written atomically
  * `max_iters_per_epoch` budget cap, `--override key=value`, CPU fallback,
    transparent hugepages off for the data workers (NAS stall fix)
  * the precipitation model and `target: residual_field` are not supported
"""
import argparse
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Subset

from utils import logging_utils
logging_utils.config_logger()
from utils.darcy_loss import LpLoss
from utils.data_loader_multifiles import GetDataset
from utils.mixer_params import Params, disable_transparent_hugepages, parse_overrides
from utils.weighted_acc_rmse import weighted_acc_torch_channels, weighted_rmse_torch_channels
from networks.mixernet import MixerNet


def append_record(path, record):
    with open(path, 'a') as stream:
        stream.write(json.dumps(record, default=float) + '\n')


def valid_indices(dataset, n_samples, steps_ahead):
    """Evenly spaced sample indices over all validation years, skipping the last
    `steps_ahead` samples of each year (the loader would predict identity there)."""
    per_year = dataset.n_samples_per_year
    candidates = [i for i in range(len(dataset)) if i % per_year < per_year - steps_ahead]
    if n_samples is None or n_samples >= len(candidates):
        return candidates
    picks = np.linspace(0, len(candidates) - 1, n_samples).round().astype(int)
    return [candidates[i] for i in picks]


def get_data_loader(params, location, distributed, train):
    dataset = GetDataset(params, location, train)
    if train:
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=True, seed=params.seed)
        else:
            sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(params.seed))
        loader_dataset = dataset
    else:
        steps_ahead = params.dt * (2 if params.two_step_training else 1)
        indices = valid_indices(dataset, params.n_valid_samples, steps_ahead)
        if distributed:  # contiguous-free sharding; sums and counts are all-reduced
            indices = indices[dist.get_rank()::dist.get_world_size()]
        loader_dataset, sampler = Subset(dataset, indices), None
    loader = DataLoader(loader_dataset, batch_size=int(params.batch_size), num_workers=params.num_data_workers,
                        shuffle=False, sampler=sampler, drop_last=train, pin_memory=torch.cuda.is_available())
    return loader, dataset, sampler


class Trainer():
  def count_parameters(self):
    return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

  def __init__(self, params, world_rank):
    self.params = params
    self.world_rank = world_rank
    # a torch.device, not torch.cuda.current_device()'s int: torch.load(map_location=int) fails
    self.device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')
    self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.distributed = dist.is_initialized()
    self.world_size = dist.get_world_size() if self.distributed else 1
    self.run_dir = params.experiment_dir

    if params.log_to_wandb:
      import wandb
      self.wandb = wandb
      config = json.loads(json.dumps(params.params, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
      wandb.init(config=config, name=params.name, group=params.group, project=params.project,
                 entity=params.entity, mode=os.getenv('WANDB_MODE', 'online'),
                 tags=['era5', params.mixer['type']], resume='allow', id=params.wandb_id)
      wandb.define_metric('valid_loss', summary='min')

    if 'precip' in params:
      raise NotImplementedError('the precipitation model is not part of the mixer comparison; use train.py')

    logging.info('rank %d, begin data loader init' % world_rank)
    self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(params, params.train_data_path, self.distributed, train=True)
    self.valid_data_loader, self.valid_dataset, _ = get_data_loader(params, params.valid_data_path, self.distributed, train=False)
    self.loss_obj = LpLoss()
    logging.info('rank %d, data loader initialized' % world_rank)

    params.crop_size_x = self.valid_dataset.crop_size_x
    params.crop_size_y = self.valid_dataset.crop_size_y
    params.img_shape_x = self.valid_dataset.img_shape_x
    params.img_shape_y = self.valid_dataset.img_shape_y

    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    self.model = MixerNet(params, img_size=(params.img_shape_x, params.img_shape_y),
                          embed_dim=params.embed_dim, depth=params.depth).to(self.device)

    if self.params.enable_nhwc:
      # NHWC: Convert model to channels_last memory format
      self.model = self.model.to(memory_format=torch.channels_last)

    if params.log_to_wandb:
      self.wandb.watch(self.model)

    if params.optimizer_type == 'FusedAdam':
      from apex import optimizers
      self.optimizer = optimizers.FusedAdam(self.model.parameters(), lr=params.lr)
    else:
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)

    self.amp_dtype = getattr(torch, params.amp_dtype)
    self.gscaler = torch.amp.GradScaler(self.device_type, enabled=params.enable_amp and self.amp_dtype == torch.float16)

    if self.distributed:
      self.model = DistributedDataParallel(self.model,
                                           device_ids=[params.local_rank] if self.device_type == 'cuda' else None,
                                           output_device=params.local_rank if self.device_type == 'cuda' else None,
                                           find_unused_parameters=True)  # AFNONet's final `norm` is unused

    self.iters = 0
    self.startEpoch = 0
    self.best_valid_loss = float('inf')
    self.gpu_hours_before = 0.0   # from earlier jobs of this run
    if params.resuming:
      logging.info("Loading checkpoint %s" % params.checkpoint_path)
      self.restore_checkpoint(params.checkpoint_path)
    if params.two_step_training:
      if params.resuming == False and params.get('pretrained', False) == True:
        logging.info("Starting from pretrained one-step model at %s" % params.pretrained_ckpt_path)
        self.restore_checkpoint(params.pretrained_ckpt_path)
        self.iters = 0
        self.startEpoch = 0
        self.best_valid_loss = float('inf')
        self.gpu_hours_before = 0.0

    self.epoch = self.startEpoch

    if params.scheduler == 'ReduceLROnPlateau':
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
    elif params.scheduler == 'CosineAnnealingLR':
      self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.max_epochs, last_epoch=self.startEpoch-1)
    else:
      self.scheduler = None

    self.log_model()

  @property
  def module(self):
    return self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

  def log_model(self):
    mixer = sum(p.numel() for b in self.module.blocks for p in b.filter.parameters())
    stats = {'mixer': self.params.mixer['type'], 'mixer_config': self.params.mixer,
             'parameters/total': sum(p.numel() for p in self.module.parameters()),
             'parameters/trainable': self.count_parameters(), 'parameters/mixer': mixer,
             'storage/parameter_bytes': sum(p.numel() * p.element_size() for p in self.module.parameters()),
             'grid': [self.module.h, self.module.w], 'world_size': self.world_size}
    if self.params.log_to_screen:
      logging.info("Number of trainable model parameters: {} (mixer: {})".format(stats['parameters/trainable'], mixer))
    if self.world_rank == 0:
      append_record(os.path.join(self.run_dir, 'model.jsonl'), stats)
      if self.params.log_to_wandb:
        self.wandb.summary.update({k: v for k, v in stats.items() if k.startswith(('parameters/', 'storage/'))})

  def train(self):
    if self.params.log_to_screen:
      logging.info("Starting Training Loop...")

    self.job_start = time.time()
    for epoch in range(self.startEpoch, self.params.max_epochs):
      if self.distributed:
        self.train_sampler.set_epoch(epoch)
      else:  # same per-epoch order whether or not the run was resumed
        self.train_sampler.generator.manual_seed(self.params.seed + epoch)

      start = time.time()
      tr_time, data_time, train_logs = self.train_one_epoch()
      valid_time, valid_logs = self.validate_one_epoch()

      if self.params.scheduler == 'ReduceLROnPlateau':
        self.scheduler.step(valid_logs['valid_loss'])
      elif self.params.scheduler == 'CosineAnnealingLR':
        self.scheduler.step()

      gpu_hours = self.gpu_hours_before + (time.time() - self.job_start) * self.world_size / 3600
      improved = valid_logs['valid_loss'] <= self.best_valid_loss
      if improved:
        self.best_valid_loss = valid_logs['valid_loss']
      if self.world_rank == 0:
        if self.params.save_checkpoint:
          #checkpoint at the end of every epoch
          self.save_checkpoint(self.params.checkpoint_path, gpu_hours=gpu_hours)
          if improved:
            self.save_checkpoint(self.params.best_checkpoint_path, gpu_hours=gpu_hours)
        record = {'epoch': self.epoch, 'step': self.iters, **train_logs,
                  'lr': self.optimizer.param_groups[0]['lr'],
                  'performance/train_epoch_seconds': tr_time + data_time,
                  'performance/train_step_seconds': tr_time, 'performance/data_seconds': data_time,
                  'performance/train_images_per_second': train_logs['steps'] * self.params.global_batch_size / (tr_time + data_time),
                  'performance/valid_seconds': valid_time,
                  'compute/cumulative_gpu_hours': gpu_hours}
        if self.device_type == 'cuda':
          record['memory/train_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
          record['memory/train_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        append_record(os.path.join(self.run_dir, 'epochs.jsonl'), record)
        append_record(os.path.join(self.run_dir, 'validation.jsonl'), {'epoch': self.epoch, 'step': self.iters, **valid_logs})
        if self.params.log_to_wandb:
          self.wandb.log({**{k: v for k, v in record.items() if k not in ('epoch', 'step')}, **valid_logs, 'epoch': self.epoch}, step=self.iters)

      if self.params.log_to_screen:
        logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        logging.info('Train loss: {}. Valid loss: {}'.format(train_logs['train_epoch_loss'], valid_logs['valid_loss']))

  def autocast(self):
    return torch.autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.params.enable_amp)

  def train_one_epoch(self):
    self.epoch += 1
    tr_time = 0
    data_time = 0
    steps = 0
    loss_sum = torch.zeros((), device=self.device)
    loss = torch.zeros((), device=self.device)
    self.model.train()
    if self.device_type == 'cuda':
      torch.cuda.reset_peak_memory_stats()

    data_start = time.time()
    for i, data in enumerate(self.train_data_loader, 0):
      if self.params.max_iters_per_epoch is not None and i >= self.params.max_iters_per_epoch:
        break
      self.iters += 1
      steps += 1
      inp, tar = map(lambda x: x.to(self.device, dtype = torch.float), data)
      if self.params.orography and self.params.two_step_training:
          orog = inp[:,-2:-1]

      if self.params.enable_nhwc:
        inp = inp.to(memory_format=torch.channels_last)
        tar = tar.to(memory_format=torch.channels_last)

      data_time += time.time() - data_start

      tr_start = time.time()

      self.model.zero_grad()
      with self.autocast():
        if self.params.two_step_training:
            gen_step_one = self.model(inp).to(self.device, dtype = torch.float)
            loss_step_one = self.loss_obj(gen_step_one, tar[:,0:self.params.N_out_channels])
            if self.params.orography:
                gen_step_two = self.model(torch.cat( (gen_step_one, orog), axis = 1)  ).to(self.device, dtype = torch.float)
            else:
                gen_step_two = self.model(gen_step_one).to(self.device, dtype = torch.float)
            loss_step_two = self.loss_obj(gen_step_two, tar[:,self.params.N_out_channels:2*self.params.N_out_channels])
            loss = loss_step_one + loss_step_two
        else:
            gen = self.model(inp).to(self.device, dtype = torch.float)
            loss = self.loss_obj(gen, tar)

      self.gscaler.scale(loss).backward()
      self.gscaler.step(self.optimizer)
      self.gscaler.update()

      loss_sum += loss.detach()
      if self.device_type == 'cuda':
        torch.cuda.synchronize()
      tr_time += time.time() - tr_start

      if self.params.log_to_wandb and self.iters % self.params.wandb_log_every == 0:
        self.wandb.log({'step_loss': loss.item()}, step=self.iters)
      data_start = time.time()

    logs = {'train_epoch_loss': loss_sum / max(steps, 1), 'train_last_loss': loss.detach()}
    if self.distributed:
      for key in sorted(logs.keys()):
        dist.all_reduce(logs[key])
        logs[key] = logs[key] / self.world_size
    logs = {k: float(v) for k, v in logs.items()}
    logs['steps'] = steps
    return tr_time, data_time, logs

  def validate_one_epoch(self):
    """Fixed subset; loss, per-channel latitude-weighted RMSE (physical units) and ACC.
    Runs on the unwrapped module, so ranks may hold different numbers of batches."""
    model = self.module
    model.eval()
    out = np.array(self.params.out_channels)
    names = [self.params.channel_names[c] for c in out]
    stds = torch.as_tensor(np.load(self.params.global_stds_path)[0, out, 0, 0], device=self.device, dtype=torch.float)
    clim = None
    if self.params.time_means_path and os.path.isfile(self.params.time_means_path):
      means = np.load(self.params.global_means_path)[0, out]
      clim = (np.load(self.params.time_means_path)[0, out, 0:self.params.img_shape_x] - means) / np.load(self.params.global_stds_path)[0, out]
      clim = torch.as_tensor(clim, device=self.device, dtype=torch.float)[None]

    C = len(out)
    sums = torch.zeros(3 + 2 * C, dtype=torch.float64, device=self.device)  # loss, l1, n, rmse[C], acc[C]
    valid_start = time.time()
    with torch.no_grad(), self.autocast():
      for i, data in enumerate(self.valid_data_loader, 0):
        inp, tar = map(lambda x: x.to(self.device, dtype = torch.float), data)
        if self.params.two_step_training:
          tar = tar[:, 0:self.params.N_out_channels]   # score the one-step forecast
        gen = model(inp).to(self.device, dtype = torch.float)
        n = inp.shape[0]
        sums[0] += self.loss_obj(gen, tar).double() * n
        sums[1] += nn.functional.l1_loss(gen, tar).double() * n
        sums[2] += n
        sums[3:3 + C] += (weighted_rmse_torch_channels(gen, tar) * stds).sum(0).double()
        if clim is not None:
          sums[3 + C:] += weighted_acc_torch_channels(gen - clim, tar - clim).sum(0).double()

        if self.params.save_valid_images and self.world_rank == 0:
          from torchvision.utils import save_image
          os.makedirs(os.path.join(self.run_dir, str(i)), exist_ok=True)
          save_image(torch.cat((gen[0, 0], torch.zeros((self.params.img_shape_x, 4), device=self.device), tar[0, 0]), axis=1),
                     os.path.join(self.run_dir, str(i), str(self.epoch) + '.png'))

    if self.distributed:
      dist.all_reduce(sums)
    count = sums[2].item()
    if count == 0:
      raise ValueError('validation processed no samples; check valid_data_path and n_valid_samples')
    sums = (sums / count).cpu().numpy()
    logs = {'valid_loss': float(sums[0]), 'valid_l1': float(sums[1]), 'n_valid_samples': int(count),
            'performance/valid_seconds': time.time() - valid_start}
    for c, name in enumerate(names):
      logs[f'valid_rmse/{name}'] = float(sums[3 + c])
      if clim is not None:
        logs[f'valid_acc/{name}'] = float(sums[3 + C + c])
    return time.time() - valid_start, logs

  def save_checkpoint(self, checkpoint_path, model=None, gpu_hours=0.0):
    if not model:
      model = self.model
    # write then rename: a walltime kill mid-write must not leave a truncated ckpt.tar
    tmp = checkpoint_path + '.tmp'
    torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(), 'mixer': self.params.mixer,
                'best_valid_loss': self.best_valid_loss, 'gpu_hours': gpu_hours}, tmp)
    os.replace(tmp, checkpoint_path)

  def restore_checkpoint(self, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
    try:
        self.model.load_state_dict(checkpoint['model_state'])
    except RuntimeError:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:] if key.startswith('module.') else 'module.' + key
            new_state_dict[name] = val
        self.model.load_state_dict(new_state_dict)
    self.iters = checkpoint['iters']
    self.startEpoch = checkpoint['epoch']
    self.best_valid_loss = checkpoint.get('best_valid_loss', float('inf'))
    self.gpu_hours_before = checkpoint.get('gpu_hours', 0.0)
    if self.params.resuming:  #restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == '__main__':
  disable_transparent_hugepages()  # before any DataLoader worker forks
  parser = argparse.ArgumentParser()
  parser.add_argument("--run_num", default='0', type=str)
  parser.add_argument("--yaml_config", default='./config/mixer_comparison.yaml', type=str)
  parser.add_argument("--config", default='afno', type=str)
  parser.add_argument("--enable_amp", action='store_true')
  parser.add_argument("--epsilon_factor", default = 0, type = float)
  parser.add_argument("--override", nargs='*', default=[], metavar='KEY=VALUE',
                      help='replace config values, e.g. max_epochs=1 "mixer={type: afno}"')
  args = parser.parse_args()

  params = Params(os.path.abspath(args.yaml_config), args.config, parse_overrides(args.override))
  params['epsilon_factor'] = args.epsilon_factor
  if 'residual_field' in params.target:
    # train.py's validation adds the input back to the (full-field) target as well
    raise NotImplementedError('target residual_field is not supported by the mixer comparison')

  params['world_size'] = int(os.environ.get('WORLD_SIZE', 1))
  params['global_batch_size'] = params.batch_size
  world_rank = 0
  local_rank = 0
  if params['world_size'] > 1:
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo', init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    world_rank = dist.get_rank()
    if params.batch_size % params['world_size']:
      raise ValueError(f"batch_size {params.batch_size} must be divisible by the {params['world_size']} ranks")
    params['batch_size'] = int(params.batch_size//params['world_size'])

  if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

  # Set up directory
  expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
  if world_rank==0:
    os.makedirs(os.path.join(expDir, 'training_checkpoints/'), exist_ok=True)
  if dist.is_initialized():
    dist.barrier()

  params['experiment_dir'] = os.path.abspath(expDir)
  params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
  params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')

  # Do not comment this line out please:
  args.resuming = True if os.path.isfile(params.checkpoint_path) else False

  params['resuming'] = args.resuming
  params['local_rank'] = local_rank
  params['enable_amp'] = args.enable_amp

  # W&B: one run per (config, run_num); a resumed job continues the same run
  params['name'] = f"era5-{params.mixer['type'].replace('_', '-')}-{args.config}-{args.run_num}"
  params['group'] = params.wandb_group
  params['project'] = params.wandb_project
  params['entity'] = params.wandb_entity
  # stable across jobs (Python's hash() is salted per process), so a resubmitted job resumes the run
  params['wandb_id'] = f"era5-{args.config}-{args.run_num}-" + hashlib.sha1(os.path.abspath(expDir).encode()).hexdigest()[:8]
  if world_rank==0:
    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
    params.log()

  params['log_to_wandb'] = (world_rank==0) and params['log_to_wandb']
  params['log_to_screen'] = (world_rank==0) and params['log_to_screen']

  params['in_channels'] = np.array(params['in_channels'])
  params['out_channels'] = np.array(params['out_channels'])
  if params.orography:
    params['N_in_channels'] = len(params['in_channels']) +1
  else:
    params['N_in_channels'] = len(params['in_channels'])

  params['N_out_channels'] = len(params['out_channels'])

  if world_rank == 0:
    with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
      yaml.safe_dump({str(k): str(v) for k, v in params.params.items()}, hpfile)

  trainer = Trainer(params, world_rank)
  trainer.train()
  logging.info('DONE ---- rank %d'%world_rank)
  if dist.is_initialized():
    dist.destroy_process_group()
