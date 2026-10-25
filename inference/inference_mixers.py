#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
# Derived from inference/inference.py (see LICENSE for the full license text and author list).

"""Mixer-comparison copy of inference/inference.py: autoregressive rollouts of a
trained MixerNet, scored against ERA5 and against persistence.

    python inference/inference_mixers.py --config tucker_spectral --run_num 0
    python inference/inference_mixers.py --config afno --weights FCN_weights_v0/backbone.ckpt --override_dir out/

Same as inference.py: initial conditions every 36 steps (9 days) through the first
out-of-sample year, `prediction_length` steps per rollout in fp32, latitude-weighted
RMSE (physical units), ACC against `time_means` (weighted and unweighted), and the
h5 datasets `rmse`, `acc`, `acc_unweighted` of shape (n_ics, prediction_length, C).

Different: the model is MixerNet built from the config's `mixer` (a checkpoint
trained with another mixer is refused); the year length is read from the file
(inference.py hard-codes 1,336/1,460); persistence (the initial condition held
fixed) is scored alongside (`rmse_persistence`, `acc_persistence`); means over
initial conditions go to inference_summary.json. Not carried over: perturbations,
daily climatology, coarse-grid metrics, land/sea masks, field dumps, n_history > 0.
"""
import argparse
import glob
import json
import logging
import os
import sys
from collections import OrderedDict
from datetime import datetime

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path[:0] = [os.path.join(HERE, '..'), os.path.join(HERE, '..', '..')]  # FourCastNet, repo root (attention/)

import h5py
import numpy as np
import torch

from utils import logging_utils
logging_utils.config_logger()
from utils.mixer_params import Params, disable_transparent_hugepages, parse_overrides
from utils.weighted_acc_rmse import (unweighted_acc_torch_channels, weighted_acc_torch_channels,
                                     weighted_rmse_torch_channels)
from networks.mixernet import MixerNet

DECORRELATION_TIME = 36  # 9 days, inference.py's spacing for z500


def load_model(model, checkpoint_file, mixer, device):
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    trained = checkpoint.get('mixer')  # absent in released FourCastNet (AFNO) checkpoints
    if trained is not None and trained != mixer:
        raise ValueError(f"{checkpoint_file} was trained with mixer {trained}, but the config asks for {mixer}")
    state = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in checkpoint['model_state'].items())
    model.load_state_dict(state, strict=True)
    return model.eval()


def setup(params, device):
    files_paths = sorted(glob.glob(params.inf_data_path + "/*.h5"))
    if not files_paths:
        raise FileNotFoundError(f"no .h5 files in {params.inf_data_path}")
    logging.info('Inference data from {}'.format(files_paths[0]))
    valid_data_full = h5py.File(files_paths[0], 'r')['fields']
    params['inference_file'] = files_paths[0]
    params['img_shape_x'] = valid_data_full.shape[2] - 1  # the training loader drops the last latitude row
    params['img_shape_y'] = valid_data_full.shape[3]

    in_channels, out_channels = np.array(params.in_channels), np.array(params.out_channels)
    if list(in_channels) != list(out_channels):
        raise ValueError('autoregressive inference needs in_channels == out_channels')
    if params.n_history != 0:
        raise NotImplementedError('n_history > 0 is not supported by inference_mixers.py')
    params['N_in_channels'] = len(in_channels) + (1 if params.orography else 0)
    params['N_out_channels'] = len(out_channels)
    params.means = np.load(params.global_means_path)[0, out_channels]  # (C, 1, 1)
    params.stds = np.load(params.global_stds_path)[0, out_channels]

    model = MixerNet(params, img_size=(params.img_shape_x, params.img_shape_y),
                     embed_dim=params.embed_dim, depth=params.depth).to(device)
    logging.info('Loading trained model checkpoint from {}'.format(params['best_checkpoint_path']))
    model = load_model(model, params['best_checkpoint_path'], params.mixer, device)
    return valid_data_full, model


def autoregressive_inference(params, ic, valid_data_full, model, device):
    dt = int(params.dt)
    prediction_length = int(params.prediction_length/dt)
    H = params.img_shape_x
    in_channels, out_channels = np.array(params.in_channels), np.array(params.out_channels)
    means, stds = params.means, params.stds

    valid_data = valid_data_full[ic:ic + prediction_length*dt:dt, in_channels, 0:H]
    valid_data = torch.as_tensor((valid_data - means)/stds).to(device, dtype=torch.float)

    m = torch.as_tensor((np.load(params.time_means_path)[0][out_channels] - means)/stds)[:, 0:H]  # climatology
    clim = m[None].to(device, dtype=torch.float)
    std = torch.as_tensor(stds[:, 0, 0]).to(device, dtype=torch.float)
    if params.orography:
      orog = torch.as_tensor(h5py.File(params.orography_path, 'r')['orog'][0:H])[None, None].to(device, dtype=torch.float)

    shape = (prediction_length, len(out_channels))
    metrics = {k: torch.zeros(shape, device=device) for k in
               ('rmse', 'acc', 'acc_unweighted', 'rmse_persistence', 'acc_persistence')}
    names = [params.channel_names[c] for c in out_channels]
    idx = names.index('z500') if 'z500' in names else 0   # the field inference.py reports
    persistence = valid_data[0:1]
    future_pred = valid_data[0:1]   # lead 0 is the initial condition itself
    with torch.no_grad():
      for i in range(prediction_length):
        if i > 0:
          inp = torch.cat((future_pred, orog), axis=1) if params.orography else future_pred
          future_pred = model(inp)
        tar = valid_data[i:i+1]
        metrics['rmse'][i] = weighted_rmse_torch_channels(future_pred, tar)[0] * std
        metrics['acc'][i] = weighted_acc_torch_channels(future_pred-clim, tar-clim)[0]
        metrics['acc_unweighted'][i] = unweighted_acc_torch_channels(future_pred-clim, tar-clim)[0]
        metrics['rmse_persistence'][i] = weighted_rmse_torch_channels(persistence, tar)[0] * std
        metrics['acc_persistence'][i] = weighted_acc_torch_channels(persistence-clim, tar-clim)[0]
        if params.log_to_screen:
          logging.info('Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}'.format(
              i, prediction_length, names[idx], metrics['rmse'][i, idx].item(), metrics['acc'][i, idx].item()))
    return {k: v.cpu().numpy() for k, v in metrics.items()}


def initial_conditions(params, n_samples, spacing):
    if params["ics_type"] == 'default':
      return list(range(0, n_samples - params.prediction_length, spacing))
    ics = []
    for date in params["date_strings"]:
      date_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
      ics.append(int((24*(date_obj.timetuple().tm_yday - 1) + date_obj.timetuple().tm_hour)/6))
    return ics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='0', type=str)
    parser.add_argument("--yaml_config", default='./config/mixer_comparison.yaml', type=str)
    parser.add_argument("--config", default='afno', type=str)
    parser.add_argument("--override_dir", default=None, type = str, help = 'Path to store inference outputs; must also set --weights arg')
    parser.add_argument("--weights", default=None, type=str, help = 'Path to model weights, for use with override_dir option')
    parser.add_argument("--ic_spacing", default=DECORRELATION_TIME, type=int, help='steps between initial conditions')
    parser.add_argument("--override", nargs='*', default=[], metavar='KEY=VALUE')
    args = parser.parse_args()
    disable_transparent_hugepages()  # each rollout loads ~3.4 GB of numpy arrays (NAS stall fix)
    params = Params(os.path.abspath(args.yaml_config), args.config, parse_overrides(args.override))
    if 'residual_field' in params.target:
      raise NotImplementedError('target residual_field is not supported by the mixer comparison')
    params['world_size'] = 1
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
      torch.cuda.set_device(0)
      torch.backends.cudnn.benchmark = True

    # Set up directory
    if args.override_dir is not None:
      assert args.weights is not None, 'Must set --weights argument if using --override_dir'
      expDir = args.override_dir
    else:
      assert args.weights is None, 'Cannot use --weights argument without also using --override_dir'
      expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    os.makedirs(expDir, exist_ok=True)

    params['experiment_dir'] = os.path.abspath(expDir)
    params['best_checkpoint_path'] = args.weights if args.override_dir is not None else os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference_out.log'))
    params.log()

    valid_data_full, model = setup(params, device)
    ics = initial_conditions(params, valid_data_full.shape[0], args.ic_spacing)
    logging.info("Inference for {} initial conditions".format(len(ics)))

    results = []
    for i, ic in enumerate(ics):
      logging.info("Initial condition {} of {}".format(i+1, len(ics)))
      results.append(autoregressive_inference(params, ic, valid_data_full, model, device))
    stacked = {k: np.stack([r[k] for r in results]) for k in results[0]}   # (n_ics, prediction_length, C)

    tag = params.get("inference_file_tag") or ""
    out_path = os.path.join(params['experiment_dir'], 'autoregressive_predictions' + tag + '.h5')
    logging.info("Saving files at {}".format(out_path))
    with h5py.File(out_path, 'w') as f:
      for key, value in stacked.items():
        f.create_dataset(key, data=value.astype(np.float32))
      f.attrs['initial_conditions'] = ics
      f.attrs['mixer'] = json.dumps(params.mixer)

    names = [params.channel_names[c] for c in params.out_channels]
    summary = {'mixer': params.mixer['type'], 'mixer_config': params.mixer, 'config': args.config,
               'checkpoint': params['best_checkpoint_path'], 'data_file': params['inference_file'],
               'initial_conditions': ics, 'lead_hours': [6 * int(params.dt) * i for i in range(stacked['rmse'].shape[1])],
               'rmse_units': 'physical (latitude weighted)', 'channels': names}
    for key, value in stacked.items():
      mean = value.mean(0)
      summary[key] = {name: mean[:, c].tolist() for c, name in enumerate(names)}
    with open(os.path.join(params['experiment_dir'], 'inference_summary' + tag + '.json'), 'w') as f:
      json.dump(summary, f, indent=1)
    logging.info('DONE')
