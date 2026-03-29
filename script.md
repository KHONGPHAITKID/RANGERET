## Train RangeRet

```bash
python3 train.py  --dataset ../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences  --data ./config/labels/semantic-kitti.yaml  --config ./config/RangeRet-semantickitti.yaml  --log ./log/rangeret_kitti  --fp16
```

## Train MambaRV

```bash
python3 train.py  --dataset ../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences  --data ./config/labels/semantic-kitti.yaml  --config ./config/MambaRV-semantickitti.yaml  --log ./log/mambarv_kitti  --fp16
```

## Optional Modular RangeRet Config

```bash
python3 train.py  --dataset ../dataset/SemanticKitti/data_odometry_velodyne/dataset/sequences  --data ./config/labels/semantic-kitti.yaml  --config ./config/RangeRet-semantickitti-modular.yaml  --log ./log/rangeret_kitti_modular  --fp16
```