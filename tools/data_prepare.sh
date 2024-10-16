DATA_ROOT=/home/qzj/datasets/nuscenes
#python tools/maptrv2/custom_nusc_map_converter.py --root-path /home/qzj/datasets/nuscenes --out-dir /home/qzj/datasets/nuscenes/custom/pkls/ --extra-tag nuscenes --version v1.0 --canbus /home/qzj/datasets/nuscenes
python tools/maptrv2/custom_nusc_map_converter.py --root-path /home/qzj/datasets/nuscenes --out-dir /home/qzj/datasets/nuscenes/custom/pkls2/ --extra-tag nuscenes --version v1.0 --canbus /home/qzj/datasets/nuscenes

python tools/maptrv2/custom_av2_map_converter.py --data-root /home/qzj/datasets/argoverse2/sensor/

#CONFIG=/home/qzj/code/MapTR/projects/configs/maptrv2/maptrv2_nusc_r50_24ep_qzj.py
#CHECKPOINT=/home/qzj/code/MapTR/ckpts/maptrv2_nusc_r50_24e.pth
#GPUS=1
#PORT=${PORT:-29505}
#
#ROOT_DIR=$PWD
#PYTHONPATH=$ROOT_DIR:$PYTHONPATH
#python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT tools/test.py $CONFIG $CHECKPOINT --launcher pytorch --eval chamfer