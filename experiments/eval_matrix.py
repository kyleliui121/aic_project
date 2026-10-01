# -*- coding: utf-8 -*-
"""
评测矩阵：N 个模型 × M 个测试场 一键跑完出报告表
==================================================
所有模型在所有测试场上统一评测（同一 imgsz、同一 conf），
输出 markdown 对比表（mAP50 / mAP50-95），直接贴进报告。

用法：
  python eval_matrix.py \
      --models 基线=runs/B0/weights/best.pt 零训练MEF=lapp_mef.pt \
      --sets 白天val=datasets/bdd_day.yaml 夜晚val=datasets/bdd_night.yaml \
      --out results/matrix.md

说明：模型名不要含 '='；测试场传 ultralytics 的 data.yaml 路径。
"""
import argparse
import datetime
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="名字=权重路径（可多个）")
    ap.add_argument("--sets", nargs="+", required=True, help="名字=data.yaml路径（可多个）")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None, help="cpu/cuda/0，缺省自动")
    ap.add_argument("--out", default="results/matrix.md")
    args = ap.parse_args()

    from ultralytics import YOLO

    models = dict(m.split("=", 1) for m in args.models)
    sets = dict(s.split("=", 1) for s in args.sets)
    table = {}   # (model, set) -> (map50, map5095)

    for mname, wpath in models.items():
        if not os.path.isfile(wpath):
            print("[跳过] 权重不存在: %s" % wpath)
            continue
        model = YOLO(wpath)
        for sname, ypath in sets.items():
            print(">>> %s × %s ..." % (mname, sname))
            kw = dict(data=ypath, imgsz=args.imgsz, conf=args.conf, verbose=False)
            if args.device:
                kw["device"] = args.device
            m = model.val(**kw)
            table[(mname, sname)] = (float(m.box.map50), float(m.box.map))
            print("    mAP50=%.4f  mAP50-95=%.4f" % table[(mname, sname)])

    # 输出 markdown
    lines = ["# 评测矩阵", "",
             "> 生成时间 %s ｜ imgsz=%d conf=%.2f" %
             (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), args.imgsz, args.conf),
             "", "## mAP50", "", "| 模型 | " + " | ".join(sets.keys()) + " |",
             "|---" * (len(sets) + 1) + "|"]
    for mname in models:
        row = ["%s" % mname]
        for sname in sets:
            v = table.get((mname, sname))
            row.append("%.1f" % (100 * v[0]) if v else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## mAP50-95", "", "| 模型 | " + " | ".join(sets.keys()) + " |",
              "|---" * (len(sets) + 1) + "|"]
    for mname in models:
        row = ["%s" % mname]
        for sname in sets:
            v = table.get((mname, sname))
            row.append("%.1f" % (100 * v[1]) if v else "—")
        lines.append("| " + " | ".join(row) + " |")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"%s|%s" % k: v for k, v in table.items()}, f, ensure_ascii=False, indent=1)
    print("\n已写出: %s" % args.out)


if __name__ == "__main__":
    main()
