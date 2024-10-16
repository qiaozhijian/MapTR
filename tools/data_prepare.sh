CUDA_VISIBLE_DEVICES=4

DATA_ROOT=/home/qzj/datasets/nuscenes
python tools/maptrv2/custom_nusc_map_converter.py --root-path ${DATA_ROOT} --out-dir ${DATA_ROOT}/custom/pkls/  --canbus ${DATA_ROOT}
python tools/maptrv2/custom_nusc_map_converter.py --root-path ${DATA_ROOT} --out-dir ${DATA_ROOT}/custom/pkls2/ --canbus ${DATA_ROOT} --keyframe

python tools/maptrv2/custom_av2_map_converter.py --data-root /home/qzj/datasets/argoverse2/sensor/

ROOT_DIR=$PWD
PYTHONPATH=$ROOT_DIR:$PYTHONPATH
GPUS=1
PORT=${PORT:-29505}

CONFIG=/home/qzj/code/MapTR/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_qzj.py
CHECKPOINT=/home/qzj/code/MapTR/ckpts/maptrv2_nusc_r50_24e.pth
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT tools/test.py $CONFIG $CHECKPOINT --launcher pytorch --eval chamfer

#CONFIG=/home/qzj/code/MapTR/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_qzj.py
#CHECKPOINT=/home/qzj/code/MapTR/ckpts/maptrv2_nusc_r50_24e.pth
#python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT tools/test.py $CONFIG $CHECKPOINT --launcher pytorch --eval chamfer