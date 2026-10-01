# experiments · 自研算法

两个为"明暗突变鲁棒感知"设计的算法，均已通过冒烟测试（2026-10-01，yolo11n + 3 场景图）。
正式训练需 BDD 数据 + 云 GPU（路线 B 拍板后执行）。

## train_lapp.py —— LA-PP 检测损失驱动的光照自适应预处理

**思想**（IA-YOLO AAAI'22 轻量复现）：冻结检测器，只训一个 ~10 万参数的小 CNN，
为每张图预测提亮参数 θ=(γ, 亮度, 对比度, 锐化)，优化目标是**检测器的检测损失**。
"为检测器提亮，而非为人眼提亮" —— 这是与 Zero-DCE（视觉损失驱动）的本质差异。

```bash
# 训练（B0 的 best.pt 出来后）
python train_lapp.py --data datasets/bdd_day/images --labels datasets/bdd_day/labels \
    --model runs/B0/weights/best.pt --epochs 3 --bs 8 --out lapp_weights.pt

# 用训好的权重增强单图（评测/演示）
python train_lapp.py --apply lapp_weights.pt --image 暗图.jpg --out enhanced.jpg
```

结构：`ParamNet`(4层小CNN) → `dip()` 可微滤波器组（幂次/仿射/锐化）→ 冻结YOLO → v8检测损失反传。
预期：评测时 `基线 vs ZeroDCE预处理 vs LA-PP预处理` 三行消融，LA-PP 应优于为人眼优化的增强。

## train_alt.py —— ALT 光照对抗训练（最坏情况增强）

**思想**（Shu et al. ICRA'21 的光照特化）：随机增强是"随机抽一种暗法"，
ALT 用 PGD 梯度上升找到让检测损失**最大**的 (γ, 亮度)，专练最坏光照。
两阶段实现，简单稳妥：

```bash
# 阶段1：搜索最坏光照并导出（本脚本）
python train_alt.py --data datasets/bdd_day/images --labels datasets/bdd_day/labels \
    --model runs/B0/weights/best.pt --steps 3 --out datasets/bdd_day_alt

# 阶段2：原始图 + 最坏光照图合并后，标准 ultralytics 训练
yolo train model=yolo11n.pt data=combined.yaml epochs=50 imgsz=640
```

冒烟实测：检测损失 15.4 → 17.1（对抗搜索确实找到了更难的暗法）。

## 与消融矩阵的对应

| 代码 | 消融行 | 定位 |
|---|---|---|
| train_lapp.py | C3（LA-PP）/ C4（+B1） | 模块创新：连续自适应参数学习 |
| train_alt.py | B1+（B1 的对抗升级） | 方法创新：随机→最坏情况 |
| （已在线上）MEF 虚拟多曝光 | C 组新行 | 零训练天花板：离散档位遍历 |

MEF→LA-PP 是"离散遍历→连续学习"的递进叙事；B1→ALT 是"随机采样→最坏情况"的递进叙事。

## 冒烟测试记录（2026-10-01）

- LA-PP：30 步端到端跑通，ParamNet 参数正常更新（γ 0.78→0.72），权重保存/加载正常
- ALT：3 张图 PGD 搜索跑通，检测损失放大 1.09 倍，最坏光照图+标签导出正常
- 注意：冒烟用 3 张图只验证"代码能跑"，数字无统计意义；正式结论等 BDD 全量
