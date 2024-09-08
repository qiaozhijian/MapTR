DATA_ROOT=/home/qzj/datasets/nuscenes
python tools/maptrv2/custom_nusc_map_converter.py --root-path /home/qzj/datasets/nuscenes --out-dir /home/qzj/datasets/nuscenes/pkls2 --extra-tag nuscenes --version v1.0 --canbus /home/qzj/datasets/nuscenes

CONFIG=/home/qzj/code/MapTR/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_qzj.py
CHECKPOINT=/home/qzj/code/MapTR/ckpts/maptrv2_nusc_r50_24e.pth
GPUS=1
PORT=${PORT:-29505}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch --eval chamfer