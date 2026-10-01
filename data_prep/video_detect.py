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
import json
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
    """行为过滤 v2（第四层）：轨迹累积证据 + 景物区域黑名单。
    阈值振荡的幻觉框会间歇出现——逐帧过滤会被"消失一帧"重置证据，
    因此证据累积后晋升为全片记忆的区域黑名单，命中即抑制；
    抑制时附带当帧内部运动校验，避免误伤从树前经过的真人。
    相机整体运动时清空全部记忆（全局运动下证据失效）。仅作用于 person 类。"""

    def __init__(self, iou_thr=0.6, min_age=4, energy_thr=2.5,
                 pan_thr=8.0, expand=0.20, miss_ttl=2):
        self.iou_thr, self.min_age = iou_thr, min_age
        self.energy_thr, self.pan_thr = energy_thr, pan_thr
        self.expand, self.miss_ttl = expand, miss_ttl
        self.prev_gray = None
        self.tracks = []   # [cls, conf, xy, energy_sum, count, miss]
        self.regions = []  # 膨胀后的景物区域框

    def _interior_energy(self, xy, prev, cur):
        x1, y1, x2, y2 = xy
        m = 0.2
        ix1, iy1 = int(x1 + (x2 - x1) * m), int(y1 + (y2 - y1) * m)
        ix2, iy2 = int(x2 - (x2 - x1) * m), int(y2 - (y2 - y1) * m)
        H, W = cur.shape
        ix1, iy1 = max(ix1, 0), max(iy1, 0)
        ix2, iy2 = min(ix2, W), min(iy2, H)
        if ix2 - ix1 < 4 or iy2 - iy1 < 4:
            return None
        return float(np.abs(cur[iy1:iy2, ix1:ix2].astype(np.int16) -
                            prev[iy1:iy2, ix1:ix2].astype(np.int16)).mean())

    def _expand_box(self, xy, W, H):
        x1, y1, x2, y2 = xy
        ex, ey = (x2 - x1) * self.expand, (y2 - y1) * self.expand
        return [max(0, x1 - ex), max(0, y1 - ey), min(W, x2 + ex), min(H, y2 + ey)]

    def update(self, dets, gray):
        if self.prev_gray is None:
            self.prev_gray = gray
            return dets
        # 相机平移检测：phase correlation（车流等场景运动不产生全局平移，
        # 不会误触发；只有相机真正晃动/平移才清空记忆）
        (dx, dy), _ = cv2.phaseCorrelate(self.prev_gray.astype(np.float32),
                                         gray.astype(np.float32))
        prev = self.prev_gray
        self.prev_gray = gray
        if abs(dx) + abs(dy) > self.pan_thr:   # 相机真实平移 → 证据失效
            self.tracks, self.regions = [], []
            return dets

        # 阶段1：轨迹匹配与证据累积（容忍 miss_ttl 帧短暂消失）
        unmatched = list(dets)
        kept_tracks = []
        for t in self.tracks:
            best, bi = None, self.iou_thr
            for d in unmatched:
                if d[0] == "person" and d[0] == t[0]:
                    v = _iou(d[2], t[2])
                    if v > bi:
                        best, bi = d, v
            if best is not None:
                unmatched.remove(best)
                e = self._interior_energy(best[2], prev, gray) or 0.0
                cnt = t[4] + 1
                esum = t[3] + e
                if cnt >= self.min_age and esum / cnt < self.energy_thr:
                    r = self._expand_box(best[2], gray.shape[1], gray.shape[0])
                    if not any(_iou(r, x) > 0.5 for x in self.regions):
                        self.regions.append(r)     # 阶段2：晋升为黑名单区域
                else:
                    kept_tracks.append([best[0], best[1], best[2], esum, cnt, 0])
            elif t[5] + 1 <= self.miss_ttl:
                kept_tracks.append([t[0], t[1], t[2], t[3], t[4], t[5] + 1])
        for d in unmatched:
            if d[0] == "person":
                kept_tracks.append([d[0], d[1], d[2], 0.0, 0, 0])
        self.tracks = kept_tracks[-30:]

        # 阶段3：黑名单抑制（附当帧运动校验，防误伤树前经过的真人）
        kept = []
        for d in dets:
            if d[0] == "person" and self.regions:
                cx, cy = (d[2][0] + d[2][2]) / 2, (d[2][1] + d[2][3]) / 2
                for r in self.regions:
                    if r[0] <= cx <= r[2] and r[1] <= cy <= r[3]:
                        e = self._interior_energy(d[2], prev, gray)
                        if e is None or e < self.energy_thr * 1.6:
                            d = None
                        break
            if d:
                kept.append(d)
        return kept


    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.regions, f)

    def load(self, path):
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                self.regions = json.load(f)
            return True
        return False


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
    ap.add_argument("--regions", default=None,
                    help="景物黑名单缓存文件：存在则加载（跳过冷启动），结束后保存")
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
    if args.regions and sf_base.load(args.regions):
        sf_ours.load(args.regions)
        print("已加载景物黑名单 %d 个区域（跳过冷启动）" % len(sf_ours.regions))
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
                last_base = sf_base.update(last_base, gray)
                last_ours = sf_ours.update(last_ours, gray)
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
    if args.regions:                     # 保存本次学到的景物黑名单（下次冷启动即生效）
        merged = sf_ours.regions + [r for r in sf_base.regions
                                    if not any(_iou(r, x) > 0.5 for x in sf_ours.regions)]
        with open(args.regions, "w", encoding="utf-8") as f:
            json.dump(merged, f)
        print("景物黑名单已保存: %s（%d 个区域）" % (args.regions, len(merged)))
    if args.show:
        cv2.destroyAllWindows()
    print("完成: %s (%d 帧)" % (args.out, done))


if __name__ == "__main__":
    main()
