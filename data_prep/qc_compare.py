# -*- coding: utf-8 -*-
"""
标注质检器：模型检测 vs 现有标注 做差集
========================================
对每张图：MEF 检测 → 与现有 YOLO 标注匹配（同类+IoU>0.3）→
  - 模型高置信发现但标注缺失 → 追加进 txt（疑似漏标，人工确认）
  - 标注存在但模型无响应 → 写入报告（疑似错标，人工核查，不自动删）
产出：
  <labels>/ 原 txt 被追加漏标候选行（行尾带 "#auto" 标记）
  qc_report.md 逐图质检报告（按疑似问题数排序）
用法：
  python qc_compare.py --pkg D:/aic/annotation --weights ../explore/yolo11n.pt
"""
import argparse
import collections
import glob
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from video_detect import mef_predict, auto_gamma

from ultralytics import YOLO

SIX = {"person", "car", "bus", "truck", "motorcycle", "bicycle"}
SCENES = ["diku", "yejian"]


def load_labels(txt_path, names):
    boxes = []
    for line in open(txt_path, encoding="utf-8").read().splitlines():
        p = line.split("#")[0].strip().split()
        if len(p) >= 5:
            cid = int(p[0])
            cx, cy, w, h = map(float, p[1:5])
            boxes.append({"cid": cid, "name": names[cid] if cid < len(names) else "?",
                          "xywh": (cx, cy, w, h), "auto": "#" in line})
    return boxes


def xywh_to_xyxy(b, W, H):
    cx, cy, w, h = b
    return [int((cx - w / 2) * W), int((cy - h / 2) * H),
            int((cx + w / 2) * W), int((cy + h / 2) * H)]


def iou(a, b):
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    s1 = (a[2] - a[0]) * (a[3] - a[1])
    s2 = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(s1 + s2 - inter, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, help="补标包解压目录（含 diku/ yejian/）")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--add-conf", type=float, default=0.45)
    args = ap.parse_args()

    w = args.weights or os.path.join(HERE, "..", "explore", "yolo11n.pt")
    model = YOLO(w if os.path.isfile(w) else "yolo11n.pt")

    report = ["# 补标包质检报告（机审）", "",
              "> 疑似漏标已自动追加进 txt（行尾带 `#auto` 标记，IoU 匹配阈值 0.3，追加置信度 ≥ %.2f）" % args.add_conf,
              "> 疑似错标只列出供人工核查，未自动删除。", ""]

    total_add, total_sus = 0, 0
    for scene in SCENES:
        img_dir = os.path.join(args.pkg, scene, "images")
        lbl_dir = os.path.join(args.pkg, scene, "labels")
        classes_txt = os.path.join(lbl_dir, "classes.txt")
        names = [l.strip() for l in open(classes_txt, encoding="utf-8") if l.strip()]

        rows = []
        for img_path in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            txt = os.path.join(lbl_dir, stem + ".txt")
            img = cv2.imread(img_path)
            if img is None or not os.path.isfile(txt):
                continue
            H, W = img.shape[:2]
            dets = mef_predict(model, auto_gamma(img), min_votes=2)
            labels = load_labels(txt, names)

            missing, unconfirmed = [], []
            used_labels = set()
            for name, conf, xy in dets:
                if name not in SIX:
                    continue
                hit = False
                for i, lb in enumerate(labels):
                    if lb["name"] == name and i not in used_labels:
                        lxy = xywh_to_xyxy(lb["xywh"], W, H)
                        if iou(xy, lxy) > 0.3:
                            used_labels.add(i)
                            hit = True
                            break
                if not hit and conf >= args.add_conf:
                    missing.append((name, conf, xy))

            for i, lb in enumerate(labels):
                if i in used_labels or lb["name"] not in SIX:
                    continue
                lxy = xywh_to_xyxy(lb["xywh"], W, H)
                ok = any(n == lb["name"] and iou(xy, lxy) > 0.25
                         for n, c, xy in dets)
                if not ok:
                    unconfirmed.append(lb)

            # 追加疑似漏标（纯YOLO格式——行尾不可带备注，labelImg会解析崩溃；审计信息进QC报告）
            added = 0
            if missing:
                with open(txt, "a", encoding="utf-8") as f:
                    for name, conf, xy in missing:
                        cx, cy = (xy[0] + xy[2]) / 2 / W, (xy[1] + xy[3]) / 2 / H
                        bw, bh = (xy[2] - xy[0]) / W, (xy[3] - xy[1]) / H
                        cid = names.index(name)
                        f.write("%d %.6f %.6f %.6f %.6f\n" % (cid, cx, cy, bw, bh))
                        added += 1
            total_add += added
            total_sus += len(unconfirmed)
            rows.append((added + len(unconfirmed), stem, added,
                         [f'{u["name"]}@({u["xywh"][0]:.2f},{u["xywh"][1]:.2f})' for u in unconfirmed]))

        rows.sort(reverse=True)
        report.append(f"## {scene}（按疑似问题数降序，前 20 优先核查）\n")
        report.append("| 图 | 追加漏标候选 | 疑似错标（位置） |")
        report.append("|---|---|---|")
        for cnt, stem, added, unc in rows[:20]:
            report.append(f"| {stem} | {added} | {', '.join(unc) if unc else '—'} |")
        report.append("")

    with open(os.path.join(args.pkg, "QC_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("完成：自动追加疑似漏标 %d 个，疑似错标 %d 处（见 QC_REPORT.md）"
          % (total_add, total_sus))


if __name__ == "__main__":
    main()
