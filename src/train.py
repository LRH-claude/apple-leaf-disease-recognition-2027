# -*- coding: utf-8 -*-
"""
第六步：AutoDL 云端训练脚本（支持 3 类主实验 / 5 类扩展实验）
在 AutoDL 的 JupyterLab 终端里依次运行：
    python 6_autodl_train.py resnet50
    python 6_autodl_train.py swin_t
    python 6_autodl_train.py mobilenet_v3_large
前提：数据已解压到 /root/autodl-tmp/data（train/val/test 三个子文件夹）
5 类扩展实验用 DATA_DIR=/root/autodl-tmp/data5 OUT_DIR=/root/autodl-tmp/output5

各模型使用完全相同的数据划分、预训练权重来源和训练策略：
ImageNet-1K 预训练 + Adam + 主干 lr=1e-5 / 分类头 lr=1e-4 + 余弦退火 + 30 轮。
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 可调参数 ----------
DATA_DIR = os.environ.get("DATA_DIR", "/root/autodl-tmp/data")
OUT_DIR = os.environ.get("OUT_DIR", "/root/autodl-tmp/output")
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_NAME", "resnet50")
EPOCHS = int(os.environ.get("EPOCHS", "30"))
BATCH_SIZE = 32
BACKBONE_LR = 1e-5   # 预训练主干：小学习率，避免破坏已有特征
HEAD_LR = 1e-4       # 新初始化的分类头：较大学习率
SEED = 42
NUM_WORKERS = 4

# 报告和终端里显示中文类别名；混淆矩阵图沿用英文，避免云端缺中文字体
CLASS_CN = {
    "alternaria_leaf_spot": "斑点落叶病",
    "brown_spot": "褐斑病",
    "frogeye_leaf_spot": "蛙眼叶斑病",
    "grey_spot": "灰斑病",
    "healthy": "健康",
    "mosaic": "花叶病",
    "powdery_mildew": "白粉病",
    "rust": "锈病",
    "scab": "黑星病",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name, num_classes):
    """返回模型和 (主干参数, 分类头参数)，用于分层学习率。"""
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head = list(m.fc.parameters())
    elif name == "swin_t":
        m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        m.head = nn.Linear(m.head.in_features, num_classes)
        head = list(m.head.parameters())
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        head = list(m.classifier[3].parameters())
    else:
        raise ValueError(f"未知模型 {name}，可选：resnet50 / swin_t / mobilenet_v3_large")

    head_ids = {id(p) for p in head}
    backbone = [p for p in m.parameters() if id(p) not in head_ids]
    return m, backbone, head


def measure_latency(model, dataset, device, n=50):
    """单张图像前向推理耗时（毫秒），不含读盘和解码。"""
    model.eval()
    images = torch.stack([dataset[i][0] for i in range(min(n, len(dataset)))])
    images = images.to(device)
    times = []
    with torch.no_grad():
        for i in range(images.size(0)):
            x = images[i:i + 1]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)
    times = times[5:] if len(times) > 10 else times   # 去掉预热
    return sum(times) / len(times)


def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch {torch.__version__} | 设备: {device}"
          f"{' | ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    print(f"模型: {MODEL_NAME} | 轮数: {EPOCHS} | 批大小: {BATCH_SIZE}")

    # ---------- 数据预处理 ----------
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_set = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_set = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=eval_tf)
    test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=eval_tf)
    CLASS_NAMES = train_set.classes
    assert CLASS_NAMES == val_set.classes == test_set.classes, "训练/验证/测试类别不一致"
    print(f"类别: {CLASS_NAMES}")
    print(f"数据量: train {len(train_set)} / val {len(val_set)} / test {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

    # ---------- 模型、优化器、学习率调度 ----------
    model, backbone_params, head_params = build_model(MODEL_NAME, len(CLASS_NAMES))
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ---------- 训练 ----------
    best_acc = 0.0
    history = {"train_loss": [], "val_acc": []}
    best_path = os.path.join(OUT_DIR, f"{MODEL_NAME}_best.pth")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch + 1}/{EPOCHS}: loss {avg_loss:.4f} | "
              f"验证准确率 {val_acc * 100:.2f}% | 主干 lr {lr_now:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print(f"训练完成！最佳验证准确率: {best_acc * 100:.2f}% | 权重: {best_path}")

    # ---------- 测试集评估 ----------
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            y_pred += model(images).argmax(1).cpu().tolist()
            y_true += labels.tolist()

    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    cn_names = [CLASS_CN.get(c, c) for c in CLASS_NAMES]
    report = classification_report(y_true, y_pred, target_names=cn_names,
                                   digits=4, zero_division=0)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASS_NAMES)), zero_division=0)

    print(f"\n测试集总体准确率: {acc * 100:.2f}%")
    print("\n分类报告（每类 精确率/召回率/F1）:\n")
    print(report)
    with open(os.path.join(OUT_DIR, f"{MODEL_NAME}_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(f"模型: {MODEL_NAME}\n测试集准确率: {acc * 100:.2f}%\n")
        f.write(f"最佳验证准确率: {best_acc * 100:.2f}%\n\n{report}")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    plt.figure(figsize=(6.4, 5.4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
    plt.title(f"{MODEL_NAME} confusion matrix (test acc {acc * 100:.2f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{MODEL_NAME}_confusion_matrix.png"), dpi=150)
    plt.close()

    # 训练曲线
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [v * 100 for v in history["val_acc"]], "r-s", markersize=3,
             label="Val accuracy")
    ax2.set_ylabel("Val accuracy (%)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    plt.title(f"{MODEL_NAME} training curves")
    fig.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{MODEL_NAME}_training_curves.png"), dpi=150)
    plt.close()

    # ---------- 汇总指标 ----------
    latency = measure_latency(model, test_set, device)
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
    metrics = {
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "backbone_lr": BACKBONE_LR,
        "head_lr": HEAD_LR,
        "seed": SEED,
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set),
        "best_val_acc": round(best_acc, 4),
        "test_acc": round(acc, 4),
        "macro_precision": round(float(p.mean()), 4),
        "macro_recall": round(float(r.mean()), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "weighted_f1": round(float(np.average(f1, weights=support)), 4),
        "inference_ms_per_image": round(latency, 2),
        "per_class": per_class,
    }
    with open(os.path.join(OUT_DIR, f"{MODEL_NAME}_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, f"{MODEL_NAME}_history.json"), "w",
              encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\n单张推理耗时: {latency:.2f} ms")
    print(f"macro-F1: {f1.mean() * 100:.2f}% | weighted-F1: "
          f"{np.average(f1, weights=support) * 100:.2f}%")
    print(f"全部结果已保存到 {OUT_DIR}/")


if __name__ == "__main__":
    main()
