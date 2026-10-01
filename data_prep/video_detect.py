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


# 路面眩光/倒影抑制的几何先验阈值（可按相机视角调）
GLARE_ASPECT = 2.2      # car 框宽高比超过此值且低置信 → 视为反光斑
GLARE_Y = 0.78          # 框中心低于画面此比例 AND 框高 < GLARE_H → 近场路面目标
GLARE_H = 0.18


def suppress_glare_fp(dets, W, H):
    """几何先验抑制湿地反光/眩光误检：
    反光斑特征 = 横向拉长 或 贴近场路面的小框；真车（含远处车）不满足。"""
    keep = []
    for name, conf, xy in dets:
        x1, y1, x2, y2 = xy
        w, h = (x2 - x1) / W, (y2 - y1) / H
        cy = ((y1 + y2) / 2) / H
        if name == "car" and w / max(h, 1e-3) > GLARE_ASPECT and conf < 0.55:
            continue
        if cy > GLARE_Y and h < GLARE_H and conf < 0.60:
            continue
        keep.append((name, conf, xy))
    return keep


class StaticSceneryFilter:
    """行为特征过滤（第四层）：区分"会动的人"与"不动的景物"。
    真实行人在画面内存在自身运动（肢体/位移）；静止相机下树干、立柱等
    景物框内像素几乎不变。仅当相机整体静止时启用评估；相机运动时自动
    跳过（全局运动与局部运动无法区分）。当前只作用于 person 类。"""

    def __init__(self, iou_thr=0.80, min_age=6, energy_thr=2.5, global_thr=3.5):
        self.iou_thr, self.min_age = iou_thr, min_age
        self.energy_thr, self.global_thr = energy_thr, global_thr
        self.prev_gray = None
        self.tracks = []   # [cls, conf, xy, energy_sum, count]

    def update(self, dets, gray):
        if self.prev_gray is None:
            self.prev_gray = gray
            return dets, []
        gdiff = float(np.abs(gray.astype(np.int16) -
                             self.prev_gray.astype(np.int16)).mean())
        self.prev_gray = gray
        if gdiff > self.global_thr:          # 相机在动 → 无法评估，全部放行
            self.tracks = []
            return dets, []

        scenery = []
        unmatched = list(dets)
        new_tracks = []
        for t in self.tracks:
            best, best_iou = None, self.iou_thr
            for d in unmatched:
                if d[0] == t[0] and d[0] == "person":
                    v = _iou(d[2], t[2])
                    if v > best_iou:
                        best, best_iou = d, v
            if best is not None:
                unmatched.remove(best)
                x1, y1, x2, y2 = best[2]
                m = 0.2                       # 收缩 20% 只看框内部（自身运动）
                ix1, iy1 = int(x1 + (x2 - x1) * m), int(y1 + (y2 - y1) * m)
                ix2, iy2 = int(x2 - (x2 - x1) * m), int(y2 - (y2 - y1) * m)
                if ix2 > ix1 and iy2 > iy1:
                    e = float(np.abs(gray[iy1:iy2, ix1:ix2].astype(np.int16) -
                                     self.prev_gray[iy1:iy2, ix1:ix2].astype(np.int16)).mean())
                else:
                    e = 0.0
                new_tracks.append([best[0], best[1], best[2],
                                   t[3] + e, t[4] + 1])
                if t[4] + 1 >= self.min_age and (t[3] + e) / (t[4] + 1) < self.energy_thr:
                    scenery.append(best)
        for d in unmatched:                   # 新轨迹从零累积
            new_tracks.append([d[0], d[1], d[2], 0.0, 0])
        self.tracks = new_tracks[-40:]
        scenery_ids = {id(s) for s in scenery}
        return [d for d in dets if not any(d is s for s in scenery)], scenery


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


def mef_predict(model, img, min_votes=2):
    """跨档投票融合（v3）：目标须在 ≥min_votes 档曝光中同时出现；
    框坐标按置信度加权平均——解决"取单档框导致位置标歪"的问题。"""
    per_level = [predict(model, gamma_img(img, g), tta=True) for g in GAMMA_LADDER]
    allb = sorted([b for lv in per_level for b in lv], key=lambda x: -x[1])
    clusters = []
    for b in allb:
        for c in clusters:
            if c[0][0] == b[0] and _iou(c[0][2], b[2]) > 0.45:
                c.append(b)
                break
        else:
            clusters.append([b])
    out = []
    for c in clusters:
        if len(c) < min_votes:
            continue
        top = c[0]
        wsum = sum(x[1] for x in c) or 1.0
        xy = [int(sum(x[2][k] * x[1] for x in c) / wsum) for k in range(4)]
        out.append((top[0], top[1], xy))
    return out


def _iou(a, b):
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    s1 = (a[2] - a[0]) * (a[3] - a[1])
    s2 = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(s1 + s2 - inter, 1)


class TemporalFilter:
    """时序一致性过滤（文献中的多帧持续性检查）：
    目标须连续 min_hits 个处理帧出现才显示；可容忍 max_age 帧短暂消失。
    车灯树影等闪现误检无法连续出现 → 被过滤；真实车辆持续在画面中 → 保留。"""

    def __init__(self, iou_thr=0.4, min_hits=2, max_age=2):
        self.iou_thr, self.min_hits, self.max_age = iou_thr, min_hits, max_age
        self.tracks = []          # [cls, conf, xy, hits, miss]

    def update(self, dets):
        unmatched = list(dets)
        for t in self.tracks:                     # 贪心 IoU 匹配
            best, best_iou = None, self.iou_thr
            for d in unmatched:
                if d[0] == t[0]:
                    v = _iou(d[2], t[2])
                    if v > best_iou:
                        best, best_iou = d, v
            if best is not None:
                unmatched.remove(best)
                t[1], t[2] = best[1], best[2]      # 更新置信度与位置
                t[3] += 1                           # hits
                t[4] = 0                            # miss 清零
            else:
                t[4] += 1                           # 本帧未出现
        for d in unmatched:                        # 新轨迹
            self.tracks.append([d[0], d[1], d[2], 1, 0])
        self.tracks = [t for t in self.tracks if t[4] <= self.max_age]
        return [(t[0], t[1], t[2]) for t in self.tracks if t[3] >= self.min_hits]


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
    ap.add_argument("--start-sec", type=float, default=0, help="从视频第几秒开始（跳着测小段用）")
    ap.add_argument("--duration", type=float, default=0, help="只处理多少秒（0=到结尾）")
    ap.add_argument("--min-hits", type=int, default=2,
                    help="时序过滤：目标须连续出现N个处理帧才显示（1=关闭）")
    ap.add_argument("--no-glare", action="store_true", help="关闭湿地反光/眩光几何抑制")
    ap.add_argument("--no-motion", action="store_true", help="关闭静止景物行为过滤")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    args.motion_filter = not args.no_motion

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
    if args.start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000)
    end_frame = int(args.duration * fps) if args.duration > 0 else 0

    n_out = 1 if args.mode in ("ours", "base") else 2
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (W, (H + 64) * n_out))
    print("输入 %dx%d @%.1ffps，模式=%s" % (W, H, fps, args.mode))

    fidx, done = 0, 0
    last_base, last_ours, last_pb, last_po = [], [], 0, 0
    tf_base = TemporalFilter(min_hits=args.min_hits)
    tf_ours = TemporalFilter(min_hits=args.min_hits)
    sf_base = StaticSceneryFilter()
    sf_ours = StaticSceneryFilter()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and fidx >= args.max_frames) \
                or (end_frame and fidx >= end_frame):
            break
        if fidx % args.every == 0:
            bright = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
            last_base = predict(model, frame)
            last_ours = mef_predict(model, frame) if args.mode != "base" else last_base
            if args.min_hits > 1:                 # 时序一致性过滤（基线与本方法都过滤）
                last_base = tf_base.update(last_base)
                last_ours = tf_ours.update(last_ours)
            if not args.no_glare:                 # 几何先验：湿地反光/眩光抑制
                last_base = suppress_glare_fp(last_base, W, H)
                last_ours = suppress_glare_fp(last_ours, W, H)
            if args.motion_filter:                # 行为过滤：静止景物(树干) vs 真人
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                last_base, sb = sf_base.update(last_base, gray)
                last_ours, so = sf_ours.update(last_ours, gray)
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
