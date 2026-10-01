# -*- coding: utf-8 -*-
"""
单图对比：一张照片 → 左右拼接对比图（左=基线红框，右=MEF零训练优化青框，各带PDI）
================================================================
用法：
  python compare_one.py --image 地库照片.jpg                # 输出 同名_compare.jpg
  python compare_one.py --image xx.jpg --out 对比结果.jpg
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from warning import compute_pdi
from video_detect import predict, mef_predict, draw

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.splitext(args.image)[0] + "_compare.jpg"

    wpath = next((p for p in [os.path.join(HERE, "..", "explore", "yolo11n.pt"), "yolo11n.pt"]
                  if os.path.isfile(p)), "yolo11n.pt")
    model = YOLO(wpath)
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit("读不到图片: %s" % args.image)

    base = predict(model, img)
    ours = mef_predict(model, img)
    bright = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
    pb = compute_pdi(bright, np.mean([b[1] for b in base]) if base else 0.15)
    po = compute_pdi(bright, np.mean([b[1] for b in ours]) if ours else 0.15)

    RED, CYAN = (80, 80, 250), (250, 200, 60)
    left = draw(img.copy(), base, RED)
    right = draw(img.copy(), ours, CYAN)

    # 顶部标题条（英文，cv2 中文需额外字体）
    bar = np.full((46, img.shape[1], 3), (12, 18, 32), np.uint8)
    cv2.putText(bar, "BASELINE yolo11n  [%d obj  PDI %.0f]" % (len(base), pb),
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    bar2 = np.full((46, img.shape[1], 3), (12, 18, 32), np.uint8)
    cv2.putText(bar2, "OURS MEF4+TTA  [%d obj  PDI %.0f]" % (len(ours), po),
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)

    out_img = np.hstack([np.vstack([bar, left]), np.vstack([bar2, right])])
    cv2.imwrite(out, out_img)
    print("基线: %d 目标, PDI %.0f | 本方法: %d 目标, PDI %.0f" % (len(base), pb, len(ours), po))
    print("对比图已保存:", os.path.abspath(out))


if __name__ == "__main__":
    main()
