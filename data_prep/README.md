# data_prep · 数据与部署工具包

## 工具一览

| 脚本 | 用途 | 状态 |
|---|---|---|
| `degrade_light.py` | 受控光照退化生成器（进隧道欠曝/出隧道过曝/运动模糊） | ✅ 实测 |
| `warning.py` | 感知降级指数 PDI（四等级预警，可逐帧计算） | ✅ 实测 |
| `extract_frames.py` | 自采视频按亮度分档抽帧 + 明暗突变帧自动检出 | ✅ 实测 |
| `video_detect.py` | 视频检测演示（双画面：基线 vs MEF 零训练优化，叠加 PDI 预警） | ✅ 实测 |
| `bdd_to_yolo.py` | BDD100K 标注转 YOLO 格式 + 白天/夜晚切分 | ⏸ 未测试（训练路线启用时再验） |

依赖：`pip install ultralytics opencv-python numpy -i https://pypi.tuna.tsinghua.edu.cn/simple`

---

## degrade_light.py — 受控光照退化生成器

模拟车载相机自动曝光（AE）在明暗突变场景的三种失效，生成 L0~L4 五级受控测试集。

```bash
# 进隧道方向（欠曝，默认）
python degrade_light.py --in val/images --labels val/labels --out test_dark --skip-l0

# 出隧道方向（过曝"白洞"）
python degrade_light.py --in val/images --labels val/labels --out test_bright --skip-l0 --direction bright

# 叠加长曝光运动模糊（AE 拉长曝光的拖影，随等级增强）
python degrade_light.py --in val/images --labels val/labels --out test_real --skip-l0 --blur

# 离线随机暗光训练增强（备选；训练侧优先在线增强）
python degrade_light.py --in train/images --labels train/labels --out train_aug --mode random --count 3
```

等级参数（固定→可复现；亮度跨度参考 CIE 88 隧道入口过渡量级）：

| 等级 | dark 方向 γ/亮度 | bright 方向 γ/亮度 | 模糊核 |
|---|---|---|---|
| L1 轻度 | 1.35 / 0.85 | 0.85 / 1.20 | 3px |
| L2 中度 | 1.80 / 0.65 | 0.70 / 1.40 | 5px |
| L3 重度 | 2.40 / 0.45 | 0.58 / 1.62 | 7px |
| L4 近全黑 | 3.00 / 0.30 | 0.48 / 1.85 | 9px |

标签（YOLO txt）原样复制到每级——退化不改框位置，标注零成本继承。

## warning.py — 感知降级指数 PDI

```python
from warning import compute_pdi, advise
pdi = compute_pdi(brightness=灰度均值0_255, conf_mean=检测置信度均值)  # 0~100
level, msg = advise(pdi)   # 正常/关注/降级/严重 + 驾驶建议
```
公式：`PDI = 100×(0.55×亮度漂移分 + 0.45×置信度衰减分)`，阈值 30/55/75，可按实车标定调整。

## extract_frames.py — 自采视频抽帧

```bash
python extract_frames.py --video diku1.mp4 --out frames_diku1 --per-band 50
```
按亮度三分位分 bright/mid/dark 三档抽帧；自动输出 `transitions.csv`（亮度变化率 Top-K 帧）
——进/出地库的关键帧，标注优先做这些。

## video_detect.py — 视频检测演示

```bash
# 双画面对比（上=基线，下=MEF4+TTA），每帧叠加 PDI 预警横幅
python video_detect.py --video diku1.mp4 --out diku1_both.mp4 --mode both
python video_detect.py --video ... --mode ours --every 3   # 长 video 提速
```

实时性基准（笔记本 CPU，1280×800）：自适应γ+单次 28ms（≈36FPS，实时档）；γ+TTA 61ms；MEF4+TTA 245ms（离线档）。

## bdd_to_yolo.py — BDD 转换（训练路线启用时使用）

```bash
python bdd_to_yolo.py --labels det_train.json --images bdd_images/ --out ../datasets
```
按 timeofday 属性切分白天/夜晚，各自独立 train/val 划分（防训练-测试污染），自动生成
ultralytics yaml 与切分统计报告。**注意：尚未在真实 BDD 数据上测试。**

---

## 与流水线的关系

```
自采视频 ──extract_frames──► 关键帧 ──人工标注──► 真实测试场 ─┐
白天图 ──degrade_light──► 受控测试集 L0~L4 ──────────────────┼─► 评测（experiments/eval_matrix.py）
                                                             └─► 演示（video_detect / 网页）
每帧 ──warning.compute_pdi──► 预警输出（四层链路第④层）
```
