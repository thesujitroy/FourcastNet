# FourCastNet mixer comparison

The Helio attention comparison, repeated on FourCastNet and ERA5. Only the token mixer
of the FourCastNet backbone changes. Patch embedding, norms, MLPs, head, data,
optimizer, schedule, precision, batch and budget are identical across runs.
Design notes are in `../docs/superpowers/specs/2026-10-08-fourcastnet-mixer-comparison-design.md`.

The upstream files (`train.py`, `networks/afnonet.py`, `inference/*.py`, `config/AFNO.yaml`)
are untouched. Upstream does not run in the NAS environment anyway: it needs apex and
ruamel, and `afnonet.py` imports a module numpy 2 removed.

## Mixers

| config | mixer | params (mixer) | forward / sample |
|---|---|---|---|
| `afno` | FourCastNet's AFNO2D, the baseline | 74.7 M (3.6 M) | 1.97 TFLOP |
| `vanilla` | global softmax attention, 12 heads | 99.4 M (28.3 M) | 12.49 TFLOP |
| `tucker_free` | `attention/basic_tucker_map.py`, L=4 p=q=32 r=128 | 81.4 M (10.3 M) | 2.36 TFLOP |
| `tucker_spectral` | `attention/spectral_tucker_map.py`, L=4 kh=kw=32 r=128 | 80.6 M (9.4 M) | 2.06 TFLOP |
| `tucker_adaptive` | `attention/adaptive_spectral_tucker.py`, + gate, latitude profile | 81.4 M (10.2 M) | 2.06 TFLOP |

Counts are at 720 x 1440, patch 8 (a 90 x 180 token grid), embed 768, depth 12, 20
channels, from FlopCounterMode with FFTs by formula. Tucker settings are the Helio
NAS ones. LS attention is not included because its implementation needs a square grid.

Every block is `x = x + mix(LN(x)); x = x + MLP(LN(x))` with `mix(n) = M(n) + n`.
AFNO2D adds the `+ n` itself; for the others `MixerBlock` adds it (`inner_skip`,
default on). With `afno`, `MixerNet` is AFNONet: same state_dict, same outputs, so
released FourCastNet checkpoints load into it.

Each mixer is initialized from its own random stream. For one `seed`, every
non-mixer weight (patch and position embeddings, MLPs, head) therefore starts
identical across the five runs. The spectral and adaptive Tucker maps run in fp32
under bf16 autocast.

## Before the first run

Set the six data paths in `config/mixer_comparison.yaml` (`train_data_path` ...
`global_stds_path`). They are placeholders under `/nobackupnfs1/sroy14/FourCastNet/h5/`.
The code expects FourCastNet's preprocessed layout: year files `train/*.h5`,
`test/*.h5`, `out_of_sample/*.h5`, each with `fields` of shape (T, 20, 721, 1440)
at 6 h, plus `global_means.npy`, `global_stds.npy` (1, 20, 1, 1) and
`time_means.npy` (1, 20, 721, 1440). If the channel order differs from
`channel_names`, fix that list too.

## Running on NAS (from `FourCastNet/`)

```bash
# cost: no data needed, run any time
qsub shell_scripts/nas_mixer_benchmark.pbs

# devel smoke run, 2 nodes, 50 steps
qsub -q gpu_devel@pbspl4 -l walltime=01:00:00 -l select=2:model=mil_a100:ncpus=64:ngpus=4:mem=480GB \
     -v CONFIG=tucker_spectral,NUM_NODES=2,MODE=devel shell_scripts/nas_mixer_train.pbs

# training: 8 nodes each, resubmitting resumes
for k in afno vanilla tucker_free tucker_spectral tucker_adaptive; do
  qsub -N fcn_$k -o pbs_logs/fcn_$k.out.temp -e pbs_logs/fcn_$k.err.temp \
       -v CONFIG=$k shell_scripts/nas_mixer_train.pbs
done

# rollouts against ERA5 and persistence, then the comparison table
qsub -W depend=afterany:<job ids> shell_scripts/nas_mixer_eval.pbs
```

Keep `max_epochs` fixed across resubmissions of a run. The cosine schedule is tied to
it, so extending a finished run would continue at learning rate 0.

## Outputs

`expts/mixer_comparison/<config>/<run>/` holds the checkpoints, `out.log`, and three record files:

- `model.jsonl`: parameter counts.
- `epochs.jsonl`: mean train loss, epoch time, images/s, peak memory and GPU hours.
- `validation.jsonl`: loss, plus per-channel latitude-weighted RMSE in physical units and ACC. It uses the same 256 evenly spaced test samples for every run and epoch.

W&B runs go to group `era5-mixer-comparison` in `nasa-impact/StR World Model`.
Rollouts write `autoregressive_predictions.h5` and `inference_summary.json`.
`results/mixer_rollouts.md` holds RMSE and ACC at 24/72/120 h for z500, t850, t2m,
u10 and v10, with persistence. `results/mixer_benchmark/seeds_<job>/summary.md` has
latency and memory as mean ± std over seeds.

## Tests

Run them on a compute node, never on the front end. They need CPU only, about 5 minutes:

```bash
qsub -I -q gpu_devel@pbspl4 -l select=1:model=mil_a100:ncpus=16:ngpus=1:mem=60GB -l walltime=00:30:00
cd /nobackupnfs1/sroy14/sujit_home/Tucker_map/FourCastNet && . shell_scripts/nas_env.sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider
```

The suite checks parity with AFNONet, identical non-mixer initial weights across
mixers, every mixer's training and state-dict round trip, bf16 behaviour, and config
controls. It also runs end-to-end training, resume, a 2-rank torchrun run and
inference on a synthetic FourCastNet-format dataset, plus the benchmark. FourCastNet
and HelioFM both have a top-level `utils` package, so the two suites run separately.
