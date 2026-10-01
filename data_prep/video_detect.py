# -*- coding: utf-8 -*-
"""
视频检测演示：输入自采视频 → 输出带检测框 + 感知预警的标注视频
================================================================
把"图片测试"升级成"视频测试"：逐帧检测、画框、计算 PDI 降级指数、
超阈值在画面上打红色预警——四层链路（检测→预警）的视频版。

用法：
  # 双画面对比（上=基线，下=MEF零训练优化），最有说服力，报告/答辩直接用
  python video_detect.py --video diku1.mp4 --out diku1_both.mp4 --mode both

  # 单画面：只要本方法
  python video_detect.py --video diku1.mp4 --out diku1_ours.mp4 --mode ours

选项：
  --every N     每 N 帧处理一次（中间帧复用上一帧结果，提速）
  --max-frames  只处理前 N 帧（测试用）
  --show        处理时弹窗预览
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from warning import compute_pdi, advise

from ultralytics import YOLO

CARE = {"car", "person", "truck", "bus", "bicycle", "motorcycle", "traffic light", "stop sign"}
GAMMA_LADDER = [1.0, 0.7, 0.45, 0.3]


def auto_gamma(img, target=0.45):
    mean = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() / 255.0
    if mean >= 0.45:
        return img
    mean = max(mean, 0.02)
    g = float(np.clip(np.log(target) / np.log(mean), 0.25, 1.0))
    table = ((np.linspace(0, 1, 256) ** g) * 255).astype(np.uint8)
    return cv2.LUT(img, table)


def gamma_img(img, g):
    table = ((np.linspace(0, 1, 256) ** g) * 255).astype(np.uint8)
    return cv2.LUT(img, table)


def predict(model, img, tta=False):
    r = model.predict(img, conf=0.35, imgsz=640, augment=tta, verbose=False, device="cpu")[0]
    out = []
    if r.boxes is not None:
        for b in r.boxes:
            name = model.names[int(b.cls)]
            if name in CARE:
                out.append((name, float(b.conf), [int(v) for v in b.xyxy[0]]))
    return out


def mef_predict(model, img):
    allb = []
    for g in GAMMA_LADDER:
        allb += predict(model, gamma_img(img, g), tta=True)
    allb.sort(key=lambda x: -x[1])
    keep = []
    for b in allb:
        _, _, xy = b
        dup = any(
            abs(min(k[2][2], xy[2]) - max(k[2][0], xy[0])) > 0 and
            _iou(k[2], xy) > 0.45 for k in keep)
        if not dup:
            keep.append(b)
    return keep


def _iou(a, b):
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    s1 = (a[2] - a[0]) * (a[3] - a[1])
    s2 = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(s1 + s2 - inter, 1)


def draw(img, boxes, color):
    for name, conf, xy in boxes:
        cv2.rectangle(img, (xy[0], xy[1]), (xy[2], xy[3]), color, 2)
        cv2.putText(img, "%s %.2f" % (name, conf), (xy[0], max(xy[1] - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, "%s %.2f" % (name, conf), (xy[0], max(xy[1] - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    return img


def overlay_status(img, pdi, tag, color):
    level, msg = advise(pdi)
    bar = np.full((64, img.shape[1], 3), (12, 18, 32), np.uint8)
    cv2.putText(bar, tag, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    txt = "PDI %.0f [%s] %s" % (pdi, level, msg)
    warn_color = (0, 0, 255) if pdi >= 55 else (160, 200, 120)
    cv2.putText(bar, txt.encode("ascii", "ignore").decode(), (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, warn_color, 2)
    # PDI 进度条
    x0, w = img.shape[1] - 260, 240
    cv2.rectangle(bar, (x0, 34), (x0 + w, 48), (60, 60, 60), -1)
    cv2.rectangle(bar, (x0, 34), (x0 + int(w * pdi / 100), 48), warn_color, -1)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["both", "ours", "base"], default="both")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    model = YOLO(os.path.join(os.environ.get("YOLO_W", ""), "yolo11n.pt")
                 if os.environ.get("YOLO_W") else
                 next(p for p in [os.path.join(HERE, "..", "explore", "yolo11n.pt"),
                                  "yolo11n.pt"] if os.path.isfile(p)))
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("打不开视频: %s" % args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_out = 1 if args.mode in ("ours", "base") else 2
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (W, (H + 64) * n_out))
    print("输入 %dx%d @%.1ffps，模式=%s" % (W, H, fps, args.mode))

    fidx, done = 0, 0
    last_base, last_ours, last_pb, last_po = [], [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and fidx >= args.max_frames):
            break
        if fidx % args.every == 0:
            bright = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
            last_base = predict(model, frame)
            last_ours = mef_predict(model, frame) if args.mode != "base" else last_base
            cb = np.mean([b[1] for b in last_base]) if last_base else 0.15
            co = np.mean([b[1] for b in last_ours]) if last_ours else 0.15
            last_pb = compute_pdi(bright, cb)
            last_po = compute_pdi(bright, co)
        fidx += 1

        if args.mode in ("both", "base"):
            top = overlay_status(draw(frame.copy(), last_base, (80, 80, 250)),
                                 last_pb, "BASELINE yolo11n", (80, 80, 250))
        if args.mode in ("both", "ours"):
            bot = overlay_status(draw(frame.copy(), last_ours, (250, 200, 60)),
                                 last_po, "OURS: MEF4+TTA", (250, 200, 60))
        out = {"both": np.vstack([top, bot]), "ours": bot, "base": top}[args.mode]
        vw.write(out)
        done += 1
        if done % 100 == 0:
            print("  %d 帧  PDI 基线=%.0f 本方法=%.0f" % (done, last_pb, last_po))
        if args.show:
            cv2.imshow("demo", cv2.resize(out, (W // 2, out.shape[0] // 2)))
            if cv2.waitKey(1) == 27:
                break
    cap.release(); vw.release()
    if args.show:
        cv2.destroyAllWindows()
    print("完成: %s (%d 帧)" % (args.out, done))


if __name__ == "__main__":
    main()
