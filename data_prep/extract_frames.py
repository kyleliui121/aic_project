# -*- coding: utf-8 -*-
"""
自采视频 → 按亮度分档抽帧 + 明暗突变检测
==========================================
给队友的"抽帧"环节自动化：
  1. 全片亮度曲线（隔帧采样）
  2. 按亮度三分位自动分档：bright / mid / dark，每档均匀抽 N 张
  3. 自动找出"明暗突变时刻"（帧间亮度变化率 Top-K）——
     这些帧就是进/出地库的关键帧，标注优先做它们！

用法：
  python extract_frames.py --video diku1.mp4 --out frames_diku1 --per-band 50
  python extract_frames.py --video yejian1.mp4 --out frames_yejian --step 3

输出：
  out/bright|mid|dark/*.jpg     分档帧
  out/report.csv                帧号,时间,亮度,档位
  out/transitions.csv           突变帧清单（重点标注对象）
"""
import argparse
import csv
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=2, help="隔几帧采样一次亮度")
    ap.add_argument("--per-band", type=int, default=50, help="每档抽多少张")
    ap.add_argument("--topk", type=int, default=8, help="报告前 K 个突变时刻")
    ap.add_argument("--min-jpg-q", type=int, default=95)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("打不开视频: %s" % args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 第一遍：亮度曲线
    records = []          # (帧号, 亮度0-255)
    fidx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if fidx % args.step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
            records.append((fidx, float(gray.mean())))
        fidx += 1
    cap.release()
    if not records:
        raise SystemExit("没有读到帧")
    brights = np.array([b for _, b in records])
    print("总帧 %d，采样 %d 帧，亮度均值 %.1f" % (fidx, len(records), brights.mean()))

    # 分档阈值（三分位）
    t1, t2 = np.percentile(brights, [33.3, 66.7])
    def band(b):
        return "bright" if b > t2 else ("dark" if b < t1 else "mid")

    # 突变检测：帧间亮度变化率
    deltas = []
    for i in range(1, len(records)):
        f0, b0 = records[i - 1]
        f1, b1 = records[i]
        if f1 > f0:
            deltas.append((abs(b1 - b0) / (f1 - f0), i))
    deltas.sort(reverse=True)
    trans_idx = sorted(set(i for _, i in deltas[:args.topk]))

    # 第二遍：抽帧
    for bd in ("bright", "mid", "dark"):
        os.makedirs(os.path.join(args.out, bd), exist_ok=True)
    picks = {}
    for bd in ("bright", "mid", "dark"):
        cand = [(f, b) for f, b in records if band(b) == bd]
        k = min(args.per_band, len(cand))
        sel = np.linspace(0, len(cand) - 1, k).astype(int) if cand else []
        picks[bd] = [cand[i][0] for i in sel]

    want = {}
    for bd, frames in picks.items():
        for f in frames:
            want[f] = bd
    for i in trans_idx:                      # 突变帧强制保留，归入 mid
        want.setdefault(records[i][0], "mid")

    cap = cv2.VideoCapture(args.video)
    fidx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx in want:
            cv2.imwrite(os.path.join(args.out, want[fidx], "f%06d.jpg" % fidx), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, args.min_jpg_q])
            saved += 1
        fidx += 1
    cap.release()

    with open(os.path.join(args.out, "report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "brightness", "band"])
        for fr, b in records:
            w.writerow([fr, round(fr / fps, 2), round(b, 1), band(b)])
    with open(os.path.join(args.out, "transitions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "frame", "time_s", "delta_per_frame"])
        for rank, (d, i) in enumerate(deltas[:args.topk], 1):
            w.writerow([rank, records[i][0], round(records[i][0] / fps, 2), round(d, 2)])

    print("分档阈值: dark<%.0f<mid<%.0f<bright" % (t1, t2))
    print("已保存 %d 帧：" % saved)
    for bd in ("bright", "mid", "dark"):
        print("  %-6s %3d 张" % (bd, len(os.listdir(os.path.join(args.out, bd)))))
    print("突变时刻 %d 个（见 transitions.csv，标注优先做这些）：" % len(trans_idx))
    for d, i in deltas[:min(5, len(deltas))]:
        print("   帧 %d (t=%.1fs) 亮度变化 %.2f/帧" % (records[i][0], records[i][0] / fps, d))


if __name__ == "__main__":
    main()
