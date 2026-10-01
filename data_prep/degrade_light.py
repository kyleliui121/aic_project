# -*- coding: utf-8 -*-
"""
受控光照退化生成器
==================
把正常光照图片批量压暗成 L0~L4 五个等级，用于：
  1. 生成【受控测试集】（默认模式）：每级参数固定、可复现，
     配合评测脚本画"光照等级-精度"曲线 —— 报告核心图之一
  2. 离线生成【暗光训练数据】（--mode random）：可选。
     训练侧更推荐 ultralytics 在线增强，此模式作为备选

依赖：pip install opencv-python numpy

目录约定（兼容 YOLO 格式）
--------------------------
输入 --in 可以是：
  a) 纯图片目录：  in/*.jpg|png
  b) YOLO数据集：  in/images/*.jpg + in/labels/*.txt
     （也支持 --labels 单独指定标签目录）
输出：
  out/L0/images + out/L0/labels
  out/L1/images + out/L1/labels   ← 标签原样复制（压暗不改变框位置）
  ...
  out/L4/...
  其中 L0 为原图拷贝（--skip-l0 可跳过，直接用原目录当 L0）

用法示例
--------
  # 1) 把白天验证集生成五档受控测试集
  python degrade_light.py --in bdd_day_val/images --labels bdd_day_val/labels --out test_degraded

  # 2) 原图就是 L0，只要暗的四档
  python degrade_light.py --in bdd_day_val/images --labels bdd_day_val/labels --out test_degraded --skip-l0

  # 3) 给训练集离线造 3 倍随机暗光增强（备选方案）
  python degrade_light.py --in train/images --labels train/labels --out train_aug --mode random --count 3 --seed 42

等级参数（固定，受控可复现）
--------------------------
  等级 | 名称     | gamma | 亮度系数 | 噪声σ | 说明
  L0   | 正常照明 |  -    |  -       |  -    | 原图
  L1   | 轻度衰减 | 1.35  | 0.85     | 3     | 阴天/傍晚
  L2   | 中度衰减 | 1.80  | 0.65     | 6     | 地库入口
  L3   | 重度衰减 | 2.40  | 0.45     | 10    | 隧道深处
  L4   | 近全黑   | 3.00  | 0.30     | 15    | 无照明路段
"""

import argparse
import os
import random
import shutil

import cv2
import numpy as np

# 每级固定参数：{等级: (gamma, brightness, noise_sigma)}
# dark 方向 = 进隧道/地库（AE未及反应，画面欠曝）
LEVELS = {
    "L1": (1.35, 0.85, 3),
    "L2": (1.80, 0.65, 6),
    "L3": (2.40, 0.45, 10),
    "L4": (3.00, 0.30, 15),
}
# bright 方向 = 出隧道"白洞"（AE过冲，画面过曝），gamma<1 提亮
LEVELS_BRIGHT = {
    "L1": (0.85, 1.20, 2),
    "L2": (0.70, 1.40, 1),
    "L3": (0.58, 1.62, 1),
    "L4": (0.48, 1.85, 0),
}
# 运动模糊核（像素）：模拟 AE 拉长曝光时间的拖影，越暗曝光越长
BLUR_KERNEL = {"L1": 3, "L2": 5, "L3": 7, "L4": 9}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def make_lut(gamma):
    """gamma>1 压暗的查找表：out = 255 * (in/255)^gamma"""
    table = (np.linspace(0, 1, 256) ** gamma) * 255.0
    return table.astype(np.uint8)


def degrade(img, gamma, brightness, noise_sigma, rng=None):
    """对一张 BGR 图执行 压暗 -> 亮度缩放 -> 加噪"""
    out = cv2.LUT(img, make_lut(gamma))
    out = np.clip(out.astype(np.float32) * brightness, 0, 255)
    if noise_sigma > 0:
        rng = rng or np.random.default_rng()
        noise = rng.normal(0, noise_sigma, out.shape).astype(np.float32)
        out = np.clip(out + noise, 0, 255)
    return out.astype(np.uint8)


def find_images(in_dir):
    if os.path.isdir(os.path.join(in_dir, "images")):
        img_dir, label_dir = os.path.join(in_dir, "images"), os.path.join(in_dir, "labels")
    else:
        img_dir, label_dir = in_dir, None
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS))
    return img_dir, files, label_dir


def copy_labels(img_dir, files, label_dir, dst_dir, used_labels):
    """把与图片同名的 .txt 标签复制到输出目录（压暗不改变框位置）"""
    if label_dir is None or not os.path.isdir(label_dir):
        return
    dst_labels = os.path.join(dst_dir, "labels")
    for f in files:
        txt = os.path.splitext(f)[0] + ".txt"
        src = os.path.join(label_dir, txt)
        if os.path.isfile(src):
            os.makedirs(dst_labels, exist_ok=True)
            shutil.copy2(src, os.path.join(dst_labels, txt))
            used_labels.add(txt)


def main():
    ap = argparse.ArgumentParser(description="受控光照退化生成器（L0~L4）")
    ap.add_argument("--in", dest="in_dir", required=True, help="输入目录（纯图片或 YOLO 数据目录）")
    ap.add_argument("--labels", default=None, help="标签目录（缺省时自动探测 in/labels）")
    ap.add_argument("--out", dest="out_dir", required=True, help="输出目录")
    ap.add_argument("--mode", choices=["fixed", "random"], default="fixed",
                    help="fixed=受控五档（默认，做测试集）；random=随机暗光（做离线增强）")
    ap.add_argument("--count", type=int, default=1, help="random 模式下每张图生成几份")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（保证可复现）")
    ap.add_argument("--skip-l0", action="store_true", help="不复制原图（原图本身即 L0）")
    ap.add_argument("--direction", choices=["dark", "bright"], default="dark",
                    help="dark=进隧道欠曝（默认）；bright=出隧道白洞过曝")
    ap.add_argument("--blur", action="store_true",
                    help="叠加运动模糊（模拟AE拉长曝光的拖影，随等级增强）")
    ap.add_argument("--quality", type=int, default=95, help="输出 jpg 质量")
    args = ap.parse_args()

    img_dir, files, auto_label_dir = find_images(args.in_dir)
    label_dir = args.labels or auto_label_dir
    if not files:
        raise SystemExit("输入目录里没有图片: %s" % img_dir)
    print("输入: %d 张图片（%s）" % (len(files), img_dir))
    print("标签: %s" % (label_dir if label_dir and os.path.isdir(label_dir) else "无"))

    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    stats = {}

    # L0：原图拷贝
    if not args.skip_l0:
        dst = os.path.join(args.out_dir, "L0", "images")
        os.makedirs(dst, exist_ok=True)
        used = set()
        copy_labels(img_dir, files, label_dir, os.path.join(args.out_dir, "L0"), used)
        for f in files:
            shutil.copy2(os.path.join(img_dir, f), os.path.join(dst, f))
        stats["L0"] = len(files)

    if args.mode == "fixed":
        # 受控五档：同等级所有图片用同一组参数 → 可复现基准
        table = LEVELS_BRIGHT if args.direction == "bright" else LEVELS
        for lv, (g, b, n) in table.items():
            blur_k = BLUR_KERNEL[lv] if args.blur else 0
            dst = os.path.join(args.out_dir, lv, "images")
            os.makedirs(dst, exist_ok=True)
            copy_labels(img_dir, files, label_dir, os.path.join(args.out_dir, lv), set())
            bright_sum = 0.0
            for f in files:
                img = cv2.imread(os.path.join(img_dir, f))
                if img is None:
                    print("  [跳过] 读取失败: %s" % f)
                    continue
                out = degrade(img, g, b, n, rng=np.random.default_rng(args.seed))
                if blur_k > 1:
                    out = cv2.blur(out, (blur_k, blur_k))
                cv2.imwrite(os.path.join(dst, f), out,
                            [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                bright_sum += float(out.mean())
            stats[lv] = (len(files), bright_sum / max(len(files), 1))
    else:
        # 随机暗光：每张每份独立采样参数 → 离线训练增强
        dst = os.path.join(args.out_dir, "random", "images")
        os.makedirs(dst, exist_ok=True)
        cnt = 0
        for f in files:
            img = cv2.imread(os.path.join(img_dir, f))
            if img is None:
                continue
            stem = os.path.splitext(f)[0]
            for k in range(args.count):
                g = py_rng.uniform(1.3, 3.0)
                b = py_rng.uniform(0.40, 0.90)
                n = py_rng.uniform(3, 12)
                out = degrade(img, g, b, n, rng=np_rng)
                name = "%s_dark%d.jpg" % (stem, k)
                cv2.imwrite(os.path.join(dst, name), out,
                            [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                txt = os.path.join(label_dir or "", stem + ".txt")
                if label_dir and os.path.isfile(txt):
                    d = os.path.join(args.out_dir, "random", "labels")
                    os.makedirs(d, exist_ok=True)
                    shutil.copy2(txt, os.path.join(d, "%s_dark%d.txt" % (stem, k)))
                cnt += 1
        stats["random"] = cnt

    # 汇总
    print("\n==== 完成 ====")
    for k, v in stats.items():
        if isinstance(v, tuple):
            print("  %-6s %4d 张 | 平均亮度 %.1f/255" % (k, v[0], v[1]))
        else:
            print("  %-6s %4d 张" % (k, v))
    print("输出目录: %s" % os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
