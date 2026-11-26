# -*- coding: utf-8 -*-
"""用本地重建的 5 类测试集（919 张）重新推理三个模型。

目的：
1. 验证本地测试集与云端 data5 测试集完全一致（准确率应能复现 metrics.json）；
2. 生成中文混淆矩阵（图4-3 ~ 图4-5）；
3. 生成错误样本示例图（图4-7）。

数据来源：
    data/test/<class>（5 类，共 919 张）
权重：
    results/metrics/{resnet50,mobilenet_v3_large,swin_t}_best.pth（权重不入库）
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results" / "metrics"
FIG_DIR = ROOT / "docs" / "figures"
OUT_DIR = ROOT / "results" / "local_eval"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 训练时 ImageFolder 的字母序（必须一致，否则权重对不上）
CLASS_ORDER = ["frogeye_leaf_spot", "healthy", "powdery_mildew", "rust", "scab"]
CLASS_ZH = {
    "frogeye_leaf_spot": "蛙眼叶斑病",
    "healthy": "健康",
    "powdery_mildew": "白粉病",
    "rust": "锈病",
    "scab": "黑星病",
}
# 论文表格/图片里的展示顺序：按测试样本量从大到小
DISPLAY_ORDER = ["scab", "frogeye_leaf_spot", "rust", "powdery_mildew", "healthy"]

# 统一的 ImageFolder 测试集布局：data/test/<class>
TEST_DIRS = {cls: ROOT / "data" / "test" / cls for cls in CLASS_ORDER}

MODELS = [
    ("resnet50", "ResNet50", 3),
    ("mobilenet_v3_large", "MobileNetV3", 4),
    ("swin_t", "Swin-T", 5),
]
CLOUD_ACC = {"resnet50": 98.15, "mobilenet_v3_large": 97.93, "swin_t": 98.80}

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
})


class ImageList(Dataset):
    def __init__(self, items, tf):
        self.items = items
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), label


def build_model(name):
    if name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 5)
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 5)
    elif name == "swin_t":
        m = models.swin_t(weights=None)
        m.head = nn.Linear(m.head.in_features, 5)
    else:
        raise ValueError(name)
    state = torch.load(RES_DIR / f"{name}_best.pth", map_location="cpu")
    m.load_state_dict(state)
    m.eval()
    return m


def collect_items():
    items = []
    for cls in CLASS_ORDER:
        label = CLASS_ORDER.index(cls)
        files = sorted(p for p in TEST_DIRS[cls].iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        items += [(p, label) for p in files]
        print(f"{cls:20s} {len(files):4d} 张")
    assert len(items) == 919, f"测试集总数 {len(items)} != 919"
    return items


def plot_cm(cm, model_zh, acc, stem):
    # 重排成论文展示顺序
    idx = [CLASS_ORDER.index(c) for c in DISPLAY_ORDER]
    cm_d = cm[np.ix_(idx, idx)]
    labels = [CLASS_ZH[c] for c in DISPLAY_ORDER]

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm_d, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(5), labels, rotation=30, ha="right")
    ax.set_yticks(range(5), labels)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"{model_zh} 混淆矩阵（测试准确率 {acc:.2f}%）", fontsize=11)
    vmax = cm_d.max()
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm_d[i, j]), ha="center", va="center",
                    fontsize=9, color="white" if cm_d[i, j] > vmax * 0.55 else "black")
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_errors(records, model_zh, stem, n=8):
    wrong = [r for r in records if r["true"] != r["pred"]]
    wrong.sort(key=lambda r: r["prob"])
    pick = wrong[:n]
    if not pick:
        print("没有错误样本，跳过错误图")
        return

    cols = 4
    rows = (len(pick) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10.5, 2.9 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), pick):
        ax.imshow(Image.open(r["path"]).convert("RGB"))
        true_zh = CLASS_ZH[r["true"]]
        pred_zh = CLASS_ZH[r["pred"]]
        ax.set_title(f"真实：{true_zh}\n预测：{pred_zh}（{r['prob'] * 100:.2f}%）",
                     fontsize=9, color="#B22222")
    fig.suptitle(f"{model_zh} 典型错误样本", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    torch.set_num_threads(8)
    items = collect_items()
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(ImageList(items, tf), batch_size=32, shuffle=False, num_workers=0)

    summary = {}
    for key, fig_name, fig_no in MODELS:
        print(f"\n===== {fig_name} =====")
        model = build_model(key)
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for images, labels in loader:
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                y_pred += logits.argmax(1).tolist()
                y_prob += probs.max(1).values.tolist()
                y_true += labels.tolist()

        acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
        print(f"本地测试准确率: {acc * 100:.2f}%  | 云端记录: {CLOUD_ACC[key]:.2f}%")
        if abs(acc * 100 - CLOUD_ACC[key]) > 0.05:
            print("!! 与云端不一致，先别用这批图，检查测试集是否同一批")

        p, r, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=range(5), zero_division=0)
        report = classification_report(
            y_true, y_pred, target_names=[CLASS_ZH[c] for c in CLASS_ORDER],
            digits=4, zero_division=0)
        print(report)

        cm = confusion_matrix(y_true, y_pred, labels=range(5))
        plot_cm(cm, fig_name, acc * 100, f"图4-{fig_no}_混淆矩阵_{fig_name}")

        records = []
        for (path, label), pred, prob in zip(items, y_pred, y_prob):
            records.append({
                "path": str(path),
                "true": CLASS_ORDER[label],
                "pred": CLASS_ORDER[pred],
                "prob": round(float(prob), 4),
            })
        with open(OUT_DIR / f"{key}_preds.json", "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=1)

        if key == "swin_t":
            plot_errors(records, fig_name, "图4-7_错误样本示例")

        summary[key] = {
            "local_acc": round(float(acc), 4),
            "cloud_acc": CLOUD_ACC[key],
            "macro_f1": round(float(f1.mean()), 4),
            "weighted_f1": round(float(np.average(f1, weights=sup)), 4),
            "per_class": {
                c: {"precision": round(float(p[i]), 4),
                    "recall": round(float(r[i]), 4),
                    "f1": round(float(f1[i]), 4),
                    "support": int(sup[i])}
                for i, c in enumerate(CLASS_ORDER)
            },
        }

    with open(OUT_DIR / "local_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n完成，输出：", FIG_DIR)


if __name__ == "__main__":
    main()
