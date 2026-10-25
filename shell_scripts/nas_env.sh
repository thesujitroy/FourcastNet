#!/usr/bin/sh
## Python stack, NCCL settings and W&B proxy for the FourCastNet mixer comparison on NAS:
## exactly the Helio runs' environment. Source it from the FourCastNet directory:
##   . shell_scripts/nas_env.sh
## HelioFM's nas_env.sh sets PYTHONPATH=$PWD:$(dirname $PWD), which from here is
## FourCastNet (networks/, utils/) plus the repo root (attention/, the Tucker maps).
. "$PWD/../HelioFM-surya_1.1/shell_scripts/nas_env.sh"
