# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train MapTR with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py 8
```

Train MapTR with 1 GPUs 
```
./tools/dist_train.sh ./projects/configs/maptrv2/maptrv2_nusc_r50_24ep_qzj.py 1
```

Eval MapTR with 8 GPUs
```
./tools/dist_test_map.sh ./projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py ./path/to/ckpts.pth 8
```

Eval MapTR with 1 GPUs
```
./tools/dist_test_map.sh ./projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py ./ckpts/maptrv2_nusc_r50_24e.pth 1
```




# Visualization 

we provide tools for visualization and benchmark under `path/to/MapTR/tools/maptr`