# -*- coding: utf-8 -*-
"""
LA-PP：Light-Adaptive Pre-Processing —— 检测损失驱动的光照自适应预处理
======================================================================
思想（IA-YOLO, AAAI'22 的轻量复现）：
  冻结检测器，只训一个 ~10 万参数的小 CNN（ParamNet），
  为每张图预测提亮参数 θ=(γ, 亮度, 对比度, 锐化强度)，
  优化目标是【检测器的检测损失】——"为检测器提亮，而非为人眼提亮"。
  训练时对输入做在线随机压暗（模拟明暗突变），让 ParamNet 学会自适应。

数据：YOLO 格式（images/ + labels/），建议 BDD 白天训练集
前置：--model 传已微调的 B0 best.pt（冒烟测试可直接用 yolo11n.pt）

用法：
  # 训练（云 GPU 建议；CPU 也能跑只是慢）
  python train_lapp.py --data datasets/bdd_day/images --labels datasets/bdd_day/labels \
      --model runs/B0/weights/best.pt --epochs 3 --bs 8 --out lapp_weights.pt

  # 训练好的权重给单图做增强（评测/演示用）
  python train_lapp.py --apply lapp_weights.pt --image dark.jpg --out enhanced.jpg
"""
import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

IMG_EXTS = (".jpg", ".jpeg", ".png")
CLS_ID = {"car": 0, "truck": 1, "person": 2, "sign": 3, "bus": 5, "motorcycle": 3}  # 冒烟用


# ---------------- ParamNet：预测滤波参数的小 CNN ----------------
class ParamNet(nn.Module):
    """输入图 → (γ, 亮度a, 对比度b, 锐化s)，全部经 sigmoid 压到有效区间"""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(32, 4)
        # 参数区间：γ∈[0.30,1.30]  a∈[0.6,1.8]  b∈[-0.25,0.25]  s∈[0,0.8]
        self.lo = torch.tensor([0.30, 0.60, -0.25, 0.00])
        self.hi = torch.tensor([1.30, 1.80, 0.25, 0.80])

    def forward(self, x):                     # x: [B,3,H,W] in [0,1]
        t = torch.sigmoid(self.head(self.backbone(x)))       # [B,4] in (0,1)
        lo = self.lo.to(x.device).unsqueeze(0)
        hi = self.hi.to(x.device).unsqueeze(0)
        return lo + (hi - lo) * t             # [B,4]


# ---------------- DIP：可微滤波器组 ----------------
_GK = None
def _gauss_kernel(k=5, sigma=1.0, device="cpu"):
    global _GK
    if _GK is None or _GK.device != torch.device(device):
        ax = torch.arange(k, dtype=torch.float32, device=device) - k // 2
        g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        kernel = (g[:, None] * g[None, :])
        kernel = (kernel / kernel.sum()).view(1, 1, k, k).repeat(3, 1, 1, 1)
        _GK = kernel
    return _GK


def dip(img, theta):
    """可微提亮：γ 幂次 → 亮度/对比度仿射 → 可调锐化。img [B,3,H,W] in [0,1]"""
    def s(i):  # θ 第 i 列 → [B,1,1,1] 便于广播
        return theta[:, i:i + 1].unsqueeze(-1).unsqueeze(-1)
    gamma, bright, contrast, sharp = s(0), s(1), s(2), s(3)
    out = torch.clamp(img, 1e-4, 1.0) ** gamma
    out = out * bright + contrast
    k = _gauss_kernel(device=img.device)
    blur = F.conv2d(out, k, padding=k.shape[-1] // 2, groups=3)
    out = out + sharp * (out - blur)
    return out.clamp(0.0, 1.0)


def degrade_torch(img, rng, p=0.7):
    """在线随机压暗（对输入做，不需要梯度）。img [B,3,H,W] in [0,1]"""
    if random.random() > p:
        return img
    g = float(rng.uniform(1.3, 3.0))
    b = float(rng.uniform(0.40, 0.90))
    out = img.clamp(1e-4, 1.0) ** g * b
    noise = torch.randn_like(out) * float(rng.uniform(3, 12)) / 255.0
    return (out + noise).clamp(0.0, 1.0)


# ---------------- 数据 ----------------
def load_yolo_folder(img_dir, label_dir, size=640):
    items = []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith(IMG_EXTS):
            continue
        stem = os.path.splitext(f)[0]
        im = cv2.imread(os.path.join(img_dir, f))
        im = cv2.resize(im, (size, size))
        labels = []
        txt = os.path.join(label_dir, stem + ".txt")
        if os.path.isfile(txt):
            for line in open(txt, encoding="utf-8"):
                p = line.split()
                if len(p) >= 5:
                    labels.append([float(v) for v in p[:5]])
        items.append((stem, im, labels))
    return items


def make_batch(items, device):
    imgs = torch.from_numpy(
        np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for _, im, _ in items])
    ).float().permute(0, 3, 1, 2).to(device) / 255.0
    batch_idx, cls, boxes = [], [], []
    for bi, (_, _, labels) in enumerate(items):
        for L in labels:
            batch_idx.append(bi); cls.append([int(L[0])]); boxes.append(L[1:5])
    batch = {
        "img": imgs,
        "batch_idx": torch.tensor(batch_idx, dtype=torch.float32, device=device),
        "cls": torch.tensor(cls, dtype=torch.float32, device=device).view(-1, 1),
        "bboxes": torch.tensor(boxes, dtype=torch.float32, device=device).view(-1, 4),
    }
    return imgs, batch


# ---------------- 主流程 ----------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备:", device)
    yolo = YOLO(args.model)
    yolo.model.to(device).eval()
    for p in yolo.model.parameters():
        p.requires_grad_(False)
    # v8DetectionLoss 需要 model.args 带三项增益（有的版本是 dict，统一换成 Namespace）
    from types import SimpleNamespace
    if not isinstance(getattr(yolo.model, "args", None), SimpleNamespace):
        yolo.model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    criterion = v8DetectionLoss(yolo.model)

    items = load_yolo_folder(args.data, args.labels)
    if not items:
        raise SystemExit("数据目录为空: %s" % args.data)
    print("训练图: %d 张" % len(items))

    pnet = ParamNet().to(device).train()
    optim = torch.optim.AdamW(pnet.parameters(), lr=args.lr)
    rng = random.Random(args.seed)
    bs = args.bs
    steps_total = args.epochs * max(1, len(items) // bs)
    step = 0
    import math
    while step < steps_total:
        random.shuffle(items)
        for i in range(0, len(items) - bs + 1, bs):
            if step >= steps_total:
                break
            chunk = items[i:i + bs]
            imgs, batch = make_batch(chunk, device)
            dark = torch.stack([degrade_torch(im.unsqueeze(0), rng)[0] for im in imgs])
            theta = pnet(dark)
            enhanced = dip(dark, theta)
            preds = yolo.model(enhanced)              # eval 模式返回 (y, x)
            loss, _ = criterion(preds, batch)
            loss = loss.sum() if torch.is_tensor(loss) else sum(loss)   # 兼容不同版本返回
            optim.zero_grad(); loss.backward(); optim.step()
            if step % 20 == 0:
                print("step %d/%d  det_loss=%.3f  γ=%.2f a=%.2f" % (
                    step, steps_total, float(loss),
                    float(theta[0, 0]), float(theta[0, 1])))
            if args.save_samples and step % 200 == 0:
                with torch.no_grad():
                    vis = (enhanced[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(args.save_samples, "lapp_%05d.jpg" % step),
                                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            step += 1
    torch.save({"pnet": pnet.state_dict()}, args.out)
    print("已保存:", args.out)


def apply(args):
    device = "cpu"
    pnet = ParamNet().to(device)
    pnet.load_state_dict(torch.load(args.apply, map_location=device)["pnet"])
    pnet.eval()
    im = cv2.imread(args.image)
    H0, W0 = im.shape[:2]
    x = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        theta = pnet(x)
        y = dip(x, theta)[0].permute(1, 2, 0).numpy()
    out = (y * 255).astype(np.uint8)
    cv2.imwrite(args.out, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print("θ =", [round(float(v), 3) for v in theta[0]], "->", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LA-PP 训练/应用")
    ap.add_argument("--data", help="训练图片目录")
    ap.add_argument("--labels", help="训练标签目录")
    ap.add_argument("--model", default="yolo11n.pt", help="冻结的检测器权重")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="lapp_weights.pt")
    ap.add_argument("--save-samples", default=None, help="目录：每隔一段存增强样例图")
    ap.add_argument("--apply", default=None, help="应用模式：权重路径")
    ap.add_argument("--image", default=None, help="应用模式：输入图")
    args = ap.parse_args()
    if args.apply:
        apply(args)
    else:
        assert args.data and args.labels, "训练模式需要 --data 与 --labels"
        train(args)
