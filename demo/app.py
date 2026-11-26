# -*- coding: utf-8 -*-
"""
第九步：网页 demo 后端（Flask + 训练好的 Swin-T，5 分类）
启动方法：PyCharm 里运行本文件，然后浏览器打开 http://127.0.0.1:5000
"""
import io
import os

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

# 权重路径：优先读环境变量，默认用仓库内 weights/swin_t_best.pth（权重不入库，见 weights/README.md）
WEIGHTS = os.environ.get("APPLE_LEAF_WEIGHTS", "weights/swin_t_best.pth")

# 英文类名 -> 中文名（页面显示用；顺序必须与训练时 ImageFolder 的字母序一致）
CLASS_ZH = {
    "frogeye_leaf_spot": "蛙眼叶斑病",
    "healthy": "健康",
    "powdery_mildew": "白粉病",
    "rust": "锈病",
    "scab": "黑星病",
}
CLASS_NAMES = list(CLASS_ZH.keys())

# 与训练时完全一致的预处理（之前学过的：预处理和模型是配套的！）
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 加载训练好的模型（读档"复活"）
model = models.swin_t(weights=None)
model.head = torch.nn.Linear(model.head.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
model.eval()
print("模型加载完成")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "apple-leaf-demo-2026")

# 演示账号（毕设 demo 固定一个账号，登录页上公开提示）
USERS = {"admin": "123456"}


@app.before_request
def require_login():
    """未登录时：页面跳转到登录页，/predict 接口返回 401。"""
    if request.endpoint in ("login", "static"):
        return None
    if "user" not in session:
        if request.path == "/predict":
            return jsonify({"error": "登录已过期，请刷新页面重新登录"}), 401
        return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("index"))
    error = ""
    username = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if USERS.get(username) == password:
            session["user"] = username
            return redirect(url_for("index"))
        error = "用户名或密码错误"
    return render_template("login.html", error=error, username=username)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def index():
    return render_template("index.html", username=session["user"])


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "未收到图片"}), 400
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    x = eval_tf(img).unsqueeze(0)              # [1, 3, 224, 224]
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]  # 5 个类别的概率
    top5 = probs.topk(5)                        # 5 个类别全部列出
    results = [
        {"name": CLASS_NAMES[i], "zh": CLASS_ZH[CLASS_NAMES[i]],
         "prob": round(p.item() * 100, 2)}
        for i, p in zip(top5.indices, top5.values)
    ]
    # 拒识机制：最高置信度低于阈值时，判定为"无法识别"（开放集识别的简单实现）
    THRESHOLD = 60.0
    top1 = results[0]["prob"]
    rejected = top1 < THRESHOLD
    return jsonify({"rejected": rejected, "top_prob": top1, "results": results})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
