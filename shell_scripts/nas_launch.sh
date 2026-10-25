#!/usr/bin/sh
## Per-node torchrun launcher for train_mixers.py, run from FourCastNet by the PBS scripts:
## locally on the head node, over ssh on the others. Positional args (env vars do not cross ssh):
##   1 node rank  2 num nodes  3 world size  4 rdzv addr  5 rdzv port  6 rdzv id
##   7 PBS job id  8 TMPDIR  9 GPUs per node  10 config name (config/mixer_comparison.yaml)
##   11 run number  12 mode: full | devel  13 optional time limit in seconds (0 = none)

export NODE_RANK=$1
export NUM_NODES=$2
export WORLD_SIZE=$3
export RDZV_ADDR=$4
export RDZV_PORT=$5
export RDZV_ID=$6
export PBS_JOBID=$7
export TMPDIR=$8
export GPUS_PER_NODE=${9:-4}
CONFIG=${10:-afno}
RUN_NUM=${11:-0}
MODE=${12:-full}
TIME_LIMIT=${13:-0}

. shell_scripts/nas_env.sh

sleep 5
echo "node $NODE_RANK of $NUM_NODES, world $WORLD_SIZE, rdzv $RDZV_ADDR:$RDZV_PORT, config $CONFIG run $RUN_NUM ($MODE), dir $PWD"

## devel: a few hundred steps into a separate exp_dir, never mixed with the real runs
OVERRIDE=
if [ "$MODE" = "devel" ]; then
  OVERRIDE="--override max_epochs=1 max_iters_per_epoch=50 n_valid_samples=64 exp_dir=$PWD/expts/mixer_comparison_devel wandb_group=era5-mixer-comparison-devel"
fi
LIMIT=
if [ "$TIME_LIMIT" -gt 0 ]; then
  LIMIT="timeout --kill-after=60 $TIME_LIMIT"
fi

$LIMIT torchrun \
  --nnodes=$NUM_NODES \
  --node-rank=$NODE_RANK \
  --nproc_per_node=$GPUS_PER_NODE \
  --rdzv_id=$RDZV_ID \
  --rdzv_endpoint="$RDZV_ADDR:$RDZV_PORT" \
  --rdzv_backend=c10d \
  train_mixers.py \
  --yaml_config config/mixer_comparison.yaml \
  --config $CONFIG \
  --run_num $RUN_NUM \
  --enable_amp \
  $OVERRIDE
