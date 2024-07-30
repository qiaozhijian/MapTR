#!/usr/bin/env bash
CONFIG=/home/qzj/code/MapTR/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_qzj.py
CHECKPOINT=/home/qzj/code/MapTR/ckpts/maptrv2_nusc_r50_24e.pth
GPUS=1
PORT=${PORT:-29505}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch --eval chamfer