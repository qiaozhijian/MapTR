DATA_ROOT=/home/qzj/datasets/nuscenes
python tools/maptrv2/custom_nusc_map_converter.py --root-path $DATA_ROOT --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus $DATA_ROOT