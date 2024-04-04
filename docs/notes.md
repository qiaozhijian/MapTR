+ test: bash tools/dist_test_map.sh
+ 修改配置文件：project/configs/maptrv2/maptrv2_nusc_r50_24ep.py
+ 修改数据集samples：mmdetection3d/mmdet3d/datasets/custom_3d.py
+ 评估结果存放在test。 nuscmap_results是矢量化结果. cls_formatted存放inference和gt，每一个地图要素被采样成100点，连带预测分数。
+ 评估：给定一个阈值，使用chamfer distance确定一个预测点是否正确。按照所有预测点的score（confidence）进行排序，可以得到一系列的precisions和recalls，从而计算AP。
+ 如果停止后，GPU仍然被占用，则运行：
```angular2html
sudo fuser -v /dev/nvidia0 | awk '{for(i=1;i<=NF;i++)print $i;}' | xargs -I {} ps -p {} -o pid,comm | grep python | awk '{print "kill -9 " $1}' | sudo sh
```
+ maptr 单卡训练
  + https://github.com/hustvl/MapTR/issues/65
  + https://github.com/hustvl/MapTR/issues/69
  + https://github.com/hustvl/MapTR/issues/48
  + https://github.com/hustvl/MapTR/issues/101