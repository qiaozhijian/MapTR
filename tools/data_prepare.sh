export CUDA_VISIBLE_DEVICES=0
export ROOT_DIR=$PWD
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
export GPUS=1
export PORT=${PORT:-3096}

#DATA_ROOT=/home/qzj/datasets/nuscenes
#python tools/maptrv2/custom_nusc_map_converter.py --root-path ${DATA_ROOT} --canbus ${DATA_ROOT} 
#python tools/maptrv2/custom_nusc_map_converter.py --root-path ${DATA_ROOT} --canbus ${DATA_ROOT} --use_all_sync_time
# export CONFIG=projects/configs/maptrv2/maptrv2_nusc_r50_24ep_all.py
# export CHECKPOINT=ckpts/maptrv2_nusc_r50_24e.pth
# python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT tools/test.py $CONFIG $CHECKPOINT --launcher pytorch --eval chamfer


# python tools/maptrv2/custom_av2_map_converter.py --data-root /home/qzj/datasets/argoverse2/sensor/
export CONFIG=projects/configs/maptrv2/maptrv2_av2_3d_r50_6ep.py
export CHECKPOINT=ckpts/maptrv2_av2_3d_r50_6ep.pth
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT tools/test.py $CONFIG $CHECKPOINT --launcher pytorch --eval chamfer
