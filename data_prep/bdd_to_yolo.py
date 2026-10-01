# -*- coding: utf-8 -*-
"""
BDD100K 标注 → YOLO 格式转换 + 白天/夜晚切分
==============================================
输入：BDD 官方检测标注 json（det_train.json / det_val.json 或 10k 版）
     + 对应图片目录
输出（--out 指向的 datasets 目录）：
    bdd_day/{images,labels}/{train,val}     ← B0/B1 训练用（白天）
    bdd_night/{images,labels}/{train,val}   ← B2 训练用 / 夜晚测试场
    bdd_day.yaml / bdd_night.yaml           ← ultralytics 数据配置（自动生成）
    split_report.md                         ← 切分统计

防数据污染设计：白天/夜晚各自独立划分 train/val，
B2 用 night/train 训练、所有模型统一在 night/val 上测试，互不重叠。

用法：
  python bdd_to_yolo.py --labels det_train.json --images bdd_images/ --out ../datasets
  python bdd_to_yolo.py --labels ... --images ... --out ../datasets --val-ratio 0.1
"""
import argparse
import json
import os
import random
import shutil

# BDD 10类 → YOLO 类别 id
CAT2ID = {
    "pedestrian": 0, "person": 0,
    "car": 1,
    "bus": 2,
    "truck": 3,
    "bike": 4,
    "motor": 5,
    "rider": 6,
    "traffic light": 7,
    "traffic sign": 8,
    "train": 9,
}
NAMES = ["person", "car", "bus", "truck", "bike", "motor", "rider", "light", "sign", "train"]

IMG_EXTS = ("", ".jpg", ".jpeg", ".png")


def find_image(images_dir, name):
    for ext in IMG_EXTS:
        p = os.path.join(images_dir, name + ext) if ext else os.path.join(images_dir, name)
        if os.path.isfile(p):
            return p
    return None


def convert_entry(entry, images_dir, out_img, out_lbl, size_from="box"):
    """一条 BDD 记录 → YOLO 图片+标签。返回 (状态, 框数)"""
    name = entry["name"]
    src = find_image(images_dir, os.path.splitext(name)[0])
    if src is None:
        return "no_image", 0
    # 读尺寸：BDD 10k 统一 1280x720，不读图更快；如尺寸异常可改 cv2 读取
    W, H = 1280, 720
    lines = []
    for lb in entry.get("labels", []):
        cat = lb.get("category")
        if cat not in CAT2ID:
            continue
        b = lb.get("box2d")
        if not b:
            continue
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        if x2 <= x1 or y2 <= y1 or x1 >= W or y1 >= H:
            continue
        x2, y2 = min(x2, W), min(y2, H)
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        w, h = (x2 - x1) / W, (y2 - y1) / H
        if w < 1e-3 or h < 1e-3:          # 过小框过滤
            continue
        lines.append("%d %.6f %.6f %.6f %.6f" % (CAT2ID[cat], cx, cy, w, h))
    stem = os.path.splitext(name)[0]
    shutil.copy2(src, os.path.join(out_img, name if name.lower().endswith(IMG_EXTS[1:]) else stem + ".jpg"))
    with open(os.path.join(out_lbl, stem + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return "ok", len(lines)


def write_yaml(path, root, names):
    with open(path, "w", encoding="utf-8") as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\nnc: %d\nnames: [%s]\n"
                % (os.path.abspath(root), len(names), ", ".join(names)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="BDD 检测标注 json 路径")
    ap.add_argument("--images", required=True, help="图片目录")
    ap.add_argument("--out", required=True, help="输出 datasets 根目录")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.labels, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("标注条目: %d" % len(data))

    buckets = {"day": [], "night": [], "dusk": []}   # dusk = dawn/dusk 合并
    for e in data:
        tod = (e.get("attributes") or {}).get("timeofday", "unknown")
        if tod == "daytime" or tod == "day":
            buckets["day"].append(e)
        elif tod == "night":
            buckets["night"].append(e)
        elif tod in ("dawn/dusk", "dawn", "dusk"):
            buckets["dusk"].append(e)

    rng = random.Random(args.seed)
    report = ["# BDD 切分报告", "", "| 集合 | 图片 | 标注框 |", "|---|---|---|"]
    for key in ("day", "night", "dusk"):
        entries = buckets[key]
        rng.shuffle(entries)
        n_val = int(len(entries) * args.val_ratio)
        splits = {"train": entries[n_val:], "val": entries[:n_val]}
        for split, ents in splits.items():
            out_img = os.path.join(args.out, "bdd_%s" % key, "images", split)
            out_lbl = os.path.join(args.out, "bdd_%s" % key, "labels", split)
            os.makedirs(out_img, exist_ok=True)
            os.makedirs(out_lbl, exist_ok=True)
            total_boxes, no_img = 0, 0
            for e in ents:
                status, n = convert_entry(e, args.images, out_img, out_lbl)
                if status == "no_image":
                    no_img += 1
                else:
                    total_boxes += n
            report.append("| %s/%s | %d | %d |" % (key, split, len(ents) - no_img, total_boxes))
            print("%s/%s: %d 图, %d 框 (缺图 %d)" % (key, split, len(ents) - no_img, total_boxes, no_img))
        write_yaml(os.path.join(args.out, "bdd_%s.yaml" % key),
                   os.path.join(args.out, "bdd_%s" % key), NAMES)

    with open(os.path.join(args.out, "split_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("\n完成。数据集: %s/bdd_day, %s/bdd_night" % (args.out, args.out))
    print("yaml 已生成：bdd_day.yaml / bdd_night.yaml / bdd_dusk.yaml")


if __name__ == "__main__":
    main()
