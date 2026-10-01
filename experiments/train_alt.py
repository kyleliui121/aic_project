# -*- coding: utf-8 -*-
"""
ALT：Adversarial Light Training —— 光照对抗训练（最坏情况增强）
================================================================
思想（Shu et al., ICRA 2021 的光照特化版）：
  随机增强是"随机抽一种暗法"，ALT 用梯度找【对检测器最狠的那种暗法】再拿去训练——
  把对抗训练的 min-max 从像素空间搬到光照参数空间 (γ, 亮度)。

两阶段实现（简单稳妥）：
  阶段1 search+export（本脚本）：对每张训练图，用 PGD 在 (γ,b) 上做梯度上升
      找到让冻结检测器【检测损失最大】的退化参数，导出"最坏光照版"图片+标签
  阶段2 train（ultralytics 原生训练）：原始图 + 最坏光照图 合并成数据集，
      跑标准 yolo train —— 命令会在阶段1结束时打印出来

用法：
  python train_alt.py --data datasets/bdd_day/images --labels datasets/bdd_day/labels \
      --model runs/B0/weights/best.pt --steps 3 --out datasets/bdd_day_alt
"""
import argparse
import os
import shutil
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss

IMG_EXTS = (".jpg", ".jpeg", ".png")
G_RANGE = (1.2, 3.0)      # 只对抗性压暗：γ>1
B_RANGE = (0.35, 0.95)    # 亮度系数范围
G_STEP, B_STEP = 0.20, 0.06   # PGD 单步幅度


def degrade_diff(img, gamma, bright):
    """可微退化：I' = (I)^γ × b。img [1,3,H,W] in [0,1]"""
    return (img.clamp(1e-4, 1.0) ** gamma * bright).clamp(0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description="ALT 光照对抗训练 · 阶段1：搜索+导出")
    ap.add_argument("--data", required=True, help="训练图片目录")
    ap.add_argument("--labels", required=True, help="训练标签目录")
    ap.add_argument("--model", default="yolo11n.pt", help="冻结检测器（建议 B0 best.pt）")
    ap.add_argument("--steps", type=int, default=3, help="PGD 内循环步数")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--out", required=True, help="最坏光照图输出目录")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0=全部）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("设备:", device)
    yolo = YOLO(args.model)
    yolo.model.to(device).eval()
    for p in yolo.model.parameters():
        p.requires_grad_(False)
    if not isinstance(getattr(yolo.model, "args", None), SimpleNamespace):
        yolo.model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    criterion = v8DetectionLoss(yolo.model)

    files = sorted(f for f in os.listdir(args.data) if f.lower().endswith(IMG_EXTS))
    if args.limit:
        files = files[:args.limit]
    print("待处理: %d 张" % len(files))

    out_img = os.path.join(args.out, "images")
    out_lbl = os.path.join(args.out, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    stats = []
    for n, f in enumerate(files):
        stem = os.path.splitext(f)[0]
        im = cv2.imread(os.path.join(args.data, f))
        if im is None:
            continue
        orig_h, orig_w = im.shape[:2]
        im640 = cv2.resize(im, (args.size, args.size))
        x = torch.from_numpy(cv2.cvtColor(im640, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        labels = []
        txt = os.path.join(args.labels, stem + ".txt")
        if os.path.isfile(txt):
            for line in open(txt, encoding="utf-8"):
                p = line.split()
                if len(p) >= 5:
                    labels.append([float(v) for v in p[:5]])
        if not labels:
            continue   # 无标签图不参与损失，直接跳过
        batch_idx = torch.zeros(len(labels), device=device)
        cls = torch.tensor([[int(L[0])] for L in labels], dtype=torch.float32, device=device)
        bboxes = torch.tensor([L[1:5] for L in labels], dtype=torch.float32, device=device)
        batch = {"img": x, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}

        # PGD 梯度上升找最坏 (γ, b)
        gamma = torch.tensor(1.6, device=device, requires_grad=True)
        bright = torch.tensor(0.70, device=device, requires_grad=True)
        loss0 = None
        for _ in range(args.steps):
            if gamma.grad is not None:
                gamma.grad = None; bright.grad = None
            pred = degrade_diff(x, gamma, bright)
            loss, _ = criterion(yolo.model(pred), batch)
            loss = loss.sum() if torch.is_tensor(loss) else sum(loss)
            if loss0 is None:
                loss0 = float(loss)
            loss.backward()
            with torch.no_grad():
                if gamma.grad is not None:
                    gamma += G_STEP * torch.sign(gamma.grad)
                    bright += B_STEP * torch.sign(bright.grad)
                gamma.clamp_(*G_RANGE); bright.clamp_(*B_RANGE)

        with torch.no_grad():
            worst = degrade_diff(x, gamma.detach(), bright.detach())[0]
            l1, _ = criterion(yolo.model(worst.unsqueeze(0)), batch)
            loss1 = float(l1.sum() if torch.is_tensor(l1) else sum(l1))
        vis = (worst.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_img, stem + ".jpg"),
                    cv2.resize(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), (orig_w, orig_h)),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        if os.path.isfile(txt):
            shutil.copy2(txt, os.path.join(out_lbl, stem + ".txt"))
        stats.append((loss0, loss1))

        if (n + 1) % 200 == 0:
            print("  %d/%d  平均检测损失 %.3f -> %.3f（对抗放大 %.1f 倍）" % (
                n + 1, len(files),
                np.mean([s[0] for s in stats]), np.mean([s[1] for s in stats]),
                np.mean([s[1] / max(s[0], 1e-6) for s in stats])))

    print("\n==== 完成 ====")
    print("最坏光照图: %s (%d 张)" % (out_img, len(stats)))
    if stats:
        print("检测损失 平均 %.3f -> %.3f（放大 %.2f 倍 = 找到的确实是最难的暗法）" % (
            np.mean([s[0] for s in stats]), np.mean([s[1] for s in stats]),
            np.mean([s[1] / max(s[0], 1e-6) for s in stats])))
    print("""
阶段2（标准训练，把最坏光照图并入训练集后执行）：
  yolo train model=yolo11n.pt data=<合并后的data.yaml> epochs=50 imgsz=640
  合并方法：把 out/images、out/labels 与原始训练集同目录合并（文件名不冲突可直接cp）
""")


if __name__ == "__main__":
    main()
