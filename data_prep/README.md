# data_prep · 数据预处理工具

## degrade_light.py — 受控光照退化生成器

把正常光照图片批量压暗成 **L0~L4 五个等级**，用于构造"光照等级-精度"曲线的受控测试集（也可离线生成暗光训练增强，备选）。

### 安装依赖（一次性）

```bash
pip install opencv-python numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 常用命令

```bash
# 场景1：把白天验证集压成五档受控测试集（报告核心实验用）
python degrade_light.py --in 白天val/images --labels 白天val/labels --out test_degraded

# 场景2：原图本身就是 L0，只生成暗的四档（省一遍拷贝）
python degrade_light.py --in 白天val/images --labels 白天val/labels --out test_degraded --skip-l0

# 场景3：给训练集离线造 3 倍随机暗光增强（备选；训练侧优先用 ultralytics 在线增强）
python degrade_light.py --in train/images --labels train/labels --out train_aug --mode random --count 3 --seed 42
```

### 输出结构

```
test_degraded/
├─ L0/images/*.jpg + L0/labels/*.txt   ← 原图（--skip-l0 时无此层）
├─ L1/images/...                        ← 轻度衰减
├─ L2/...                               ← 中度衰减
├─ L3/...                               ← 重度衰减
└─ L4/...                               ← 近全黑
```

标签（YOLO txt）**原样复制**到每个等级——压暗不改变框的位置，所以标注零成本继承。
评测时把每个等级目录分别喂给 `yolo val` 即可得到五档精度，连成曲线。

### 等级参数（固定 → 可复现，写报告可直接引用）

| 等级 | 名称 | gamma | 亮度系数 | 噪声σ | 对应场景 |
|---|---|---|---|---|---|
| L0 | 正常照明 | — | — | — | 原图 |
| L1 | 轻度衰减 | 1.35 | 0.85 | 3 | 阴天/傍晚 |
| L2 | 中度衰减 | 1.80 | 0.65 | 6 | 地库入口 |
| L3 | 重度衰减 | 2.40 | 0.45 | 10 | 隧道深处 |
| L4 | 近全黑 | 3.00 | 0.30 | 15 | 无照明路段 |

> random 模式参数范围：gamma∈[1.3,3.0]，亮度∈[0.40,0.90]，噪声∈[3,12]，`--seed` 保证可复现。

### 与整条流水线的关系

```
BDD白天图 → [本脚本 L0-L4] → 受控测试集 ──┐
                                           ├→ yolo val 各档评测 → 光照-精度曲线 → 报告
自采地库帧 → 人工标注 → 真实暗光测试集 ───┘
                                           └→ 检测结果 → demo/scripts/convert_yolo_to_demo.py → 演示网页
```
