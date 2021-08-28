# SVDD-Pytorch-4-HRRP-Radar-Target-Recognition
Pytorch实现的基于SVDD的一维高分辨率雷达距离像目标识别/Target recognition of one-dimensional high-resolution radar range profile based on SVDD realized by pytorch

## 运行指令
```bash
cd SVDD-Pytorch-4-HRRP-Radar-Target-Recognition
python trainer.py -b /root/svdd_torch/seafood_svdd -c config/config_default.yaml -m test
```
指令介绍：
- -b 工作路径
- -c 配置文件路径
- -m 本次调制名称（模型命名）
- -c （如果有的话）选择checkpoint模型
