# 模型权重

模型权重（`.pth`）体积较大（Swin-T 约 105 MB），**未提交到本仓库**。

运行 demo 或评估前，请把权重放到这里：

```
weights/
└── swin_t_best.pth          # demo 默认读取此文件
```

评估脚本 `src/evaluate.py` 读取 `results/metrics/` 下的：

```
results/metrics/resnet50_best.pth
results/metrics/mobilenet_v3_large_best.pth
results/metrics/swin_t_best.pth
```

## 如何获得权重

1. 按 README 用 `src/train.py` 自己训练，训练完成会保存 `*_best.pth`；
2. 或使用本项目训练好的权重（如需要可通过作者邮箱联系）。

demo 也可用环境变量指定权重路径：

```bash
APPLE_LEAF_WEIGHTS=D:/path/to/swin_t_best.pth python demo/app.py
```
