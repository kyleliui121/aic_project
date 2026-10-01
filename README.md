# AIC 智能驾驶算法仓库

> 面向隧道及地库出入口**明暗突变场景**的鲁棒视觉感知方法
> —— 2026 第八届全球校园人工智能算法精英大赛（AIC）· 算法主题赛（智能驾驶）
>
> 本仓库包含全部**算法实现与工具链**：受控退化基准、零训练推理优化、
> 训练侧改进（LA-PP / ALT）、感知降级预警与统一评测框架。
> 设计动机与完整论证见技术报告（另行提交）。

## 1 问题背景

车载相机在隧道口、地库出入口经历 1~3 秒的自动曝光（AE）收敛窗口，
画面欠曝（"黑洞"）或过曝（"白洞"），前向目标检测在此窗口内大量漏检。
人因工程领域对此已有量化研究（驾驶员瞳孔适应、CIE 88 隧道照明规范），
但**机器视觉侧的"黑洞效应"缺乏系统评估与缓解方法**——本仓库构建之。

实测基线（COCO 预训练 YOLO11n，零样本）：受控光照衰减下关注类检出数
**17 → 16 → 8 → 1 → 0**（L0→L4）；真实地库视角图像全等级零检出。

## 2 仓库结构

```
aic_project/
├── data_prep/                 # 数据与部署工具包
│   ├── degrade_light.py       # 受控光照退化生成器（欠曝/过曝/运动模糊，L0~L4）
│   ├── extract_frames.py      # 自采视频按亮度分档抽帧 + 明暗突变帧自动检出
│   ├── warning.py             # 感知降级指数 PDI（四等级预警）
│   ├── video_detect.py        # 视频检测演示（双画面：基线 vs MEF，叠加 PDI）
│   ├── compare_one.py         # 单图左右对比（基线 vs MEF）
│   └── bdd_to_yolo.py         # BDD100K 标注 → YOLO 格式 + 昼夜切分
├── experiments/               # 算法实现
│   ├── train_lapp.py          # LA-PP：检测损失驱动的光照自适应预处理
│   ├── train_alt.py           # ALT：光照对抗训练（最坏情况增强）
│   └── eval_matrix.py         # N 模型 × M 测试场一键评测出表
└── README.md
```

依赖：`pip install ultralytics opencv-python numpy`（Python ≥ 3.10）
所有推理在普通笔记本 CPU 即可运行（详见 §4.5 实时性）。

## 3 算法实现

### 3.1 受控光照退化基准（`degrade_light.py`）

**做什么**：把正常光照图像批量生成 L0~L4 五级受控退化测试集，构成
"光照等级—检测精度"曲线的横轴（评测的可复现标尺）。

**基于什么**：模拟 AE 相机的三种真实失效模式——

| 退化轴 | 物理机理 | 参数 |
|---|---|---|
| 欠曝 dark | 进隧道，AE 反应不及 | γ∈[1.35,3.0] × 亮度[0.85,0.30] + 噪声 |
| 过曝 bright | 出隧道，AE 过冲（白洞） | γ∈[0.85,0.48] × 亮度[1.20,1.85] |
| 运动模糊 blur | AE 拉长曝光补偿 | 模糊核 3→9px 随等级 |

等级亮度跨度参考 CIE 88 隧道入口段亮度过渡的量级 [12]。
同级固定参数保证**可复现**；退化不改变标注几何，标注零成本继承。

### 3.2 零训练推理优化（`video_detect.py` / `compare_one.py`）

不改任何模型参数，纯推理侧算法。按处理顺序：

**(1) 自适应 γ 提亮** — 基于自动曝光算法：按图像均值亮度解 log 方程
求补偿 γ（目标均值 0.45），亮图不动、暗图自动加大力度。

**(2) MEF 虚拟多曝光融合** — 灵感来自车载 HDR 多曝光融合 [14] 与
Weighted Box Fusion [16] 的模型集成数学，将"多模型融合"扩展为
"多虚拟曝光融合"：

```
同一暗帧 → 5 档虚拟曝光（γ∈{1.0,0.7,0.45,0.3} + 局部自适应γ）
        → 各档独立检测（仅最佳档开 TTA，去冗余）
        → 位置聚类（IoU>0.45 或 IoS>0.65，跨类别）
        → 簇内多数票定类别、置信度加权平均定坐标（WBF）
```

- **IoS 碎片合并**：夜间一辆车常碎成"车窗框+车身框"，IoU≈0.4 逃过
  NMS，但互相包含（IoS≈1.0）——合并之（修"一车多框"）
- **跨类多数票**：car/truck 分裂票按簇内置信度加权投票定类（修类别抖动）
- **局部自适应 γ**：分块各自计算提亮曲线再平滑拼接，同帧内
  "亮便利店 + 暗路面"各得其所（修全局 γ 的混合光照盲区）

实测（受控测试，3 场景×5 级）：关注类检出 42 → 84（+100%），
此前所有免训练手段全灭的 L4 近全黑区恢复 6 个检出。

**(3) 时序一致性过滤** — 基于夜间检测文献的多帧持续性检查 [15]：
目标须连续 min_hits 个处理帧出现才显示；真实目标跨帧稳定，
灯影幻觉单帧闪现，被过滤。

**(4) 反光/眩光几何抑制** — 基于路面反射的成像几何：反光斑横向拉长
（长宽比>2.2）或贴近近场路面（中心 y>0.78 且框高<0.18）且低置信 → 抑制。
依赖相机安装几何，换装机位需重调（`--no-glare` 关闭）。

**诚实边界**：模型认知类幻觉（树干→人、发光店面→公交车）源于预训练
分布的夜间盲区，**后处理在原理上无法根治**，需训练侧方案（§3.3）。
实测发现与机理分析详见技术报告失败案例分析章节。

### 3.3 训练侧改进（`experiments/`）

**(1) LA-PP —— 检测损失驱动的光照自适应预处理**（`train_lapp.py`）

思想来自 IA-YOLO [1]（DIP 可微滤波 + CNN-PP 参数预测），轻量复现：

```
暗图 → ParamNet（4层小CNN，~10万参数）→ θ=(γ,亮度,对比度,锐化)
     → 可微滤波器组 → 冻结的 YOLO11n → 检测损失仅回传 ParamNet
```

与 Zero-DCE [3] 等视觉损失驱动增强的本质差异：**为检测器提亮，
而非为人眼提亮**。训练时对输入施加 §3.1 随机退化模拟明暗突变。
定位：MEF 是"离散档位遍历"的零训练上限，LA-PP 是其"连续参数学习"
的改进方向——两者构成递进消融。

**(2) ALT —— 光照对抗训练**（`train_alt.py`）

将对抗训练的 min-max 框架 [8] 从像素空间迁移到**光照参数空间**：
训练前用 PGD 梯度上升为每张图搜索使检测损失最大的 (γ, 亮度)——
"最坏光照"，再并入训练集做标准微调。随机增强升级为最坏情况增强。
两阶段实现（搜索导出 + ultralytics 原生训练），稳妥可控。

**(3) 感知降级预警 PDI**（`warning.py`）

借鉴 OOD 检测思想：与其在不可靠时硬输出，不如主动报告不可靠。

```
PDI = 100 × (0.50×亮度漂移分 + 0.35×置信度衰减分 + 0.15×时序不稳定分)
等级：正常(<30) / 关注(<55) / 降级(<75) / 严重(≥75) → 驾驶建议输出
```

时序不稳定分 = 检测中被上帧轨迹延续的比例（忽闪 = 不可靠）。
每帧可算、代价可忽略，直接驱动 FCW 链路的报警输出层。

### 3.4 评测框架（`eval_matrix.py`）

N 模型 × M 测试场统一评测（同 imgsz/conf），输出 mAP50 / mAP50-95
markdown 对比表。设计上的防污染原则：**训练内的"暗"可用仿真
（在线随机退化），测试内的"暗"必须是真实数据**（BDD 夜间 val、
EXDark [5]、自建地库集）。

### 3.5 实时性（笔记本 CPU 实测，1280×800）

| 配置 | 单帧 | 帧率 | 定位 |
|---|---|---|---|
| 自适应γ + 单次推理 | 28 ms | ≈36 FPS | 实时部署 |
| γ + TTA | 61 ms | ≈16 FPS | 准实时 |
| MEF 完整版 | ~326 ms | ≈3 FPS | 离线分析 |

## 4 复现指南

```bash
# 受控测试集生成
python data_prep/degrade_light.py --in val/images --labels val/labels --out test_dark --skip-l0

# 单图对比 / 视频双画面
python data_prep/compare_one.py --image 暗光照片.jpg
python data_prep/video_detect.py --video 行车视频.mp4 --out out.mp4 --mode both

# 训练侧（需 BDD 数据 + GPU）
python experiments/train_lapp.py --data ... --labels ... --model best.pt
python experiments/train_alt.py  --data ... --labels ... --model best.pt --out alt_data

# 统一评测
python experiments/eval_matrix.py --models 基线=B0.pt 本方法=C4.pt --sets 夜晚=bdd_night.yaml ...
```

## 参考文献（已逐条在线核验，2026-10）

[1] Liu W. et al. Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions. AAAI 2022.（官方代码 github.com/wenyyu/Image-Adaptive-YOLO）
[2] Yin X. et al. PE-YOLO: Pyramid Enhancement Network for Dark Object Detection. ICANN 2023. arXiv:2307.10953.（代码 github.com/XiangchenYin/PE-YOLO）
[3] Guo C. et al. Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement. CVPR 2020.
[4] Ma L. et al. Toward Fast, Flexible, and Robust Low-Light Image Enhancement. CVPR 2022.
[5] Loh Y.P. & Chan C.S. Getting to Know Low-light Images with the Exclusively Dark Dataset. Computer Vision and Image Understanding, 178:30–42, 2019.
[6] Loh Y.P. & Chan C.S. Low-light Image Enhancement using Gaussian Process for Features Retrieval. Signal Processing: Image Communication, 74:175–190, 2019.
[7] Wang D. et al. Tent: Fully Test-Time Adaptation by Entropy Minimization. ICLR 2021. arXiv:2006.10726.
[8] Shu M. et al. Adversarial Differentiable Data Augmentation for Autonomous Systems. ICRA 2021.
[9] Yang Y. & Soatto S. FDA: Fourier Domain Adaptation for Semantic Segmentation. CVPR 2020.
[10] Liu S. et al. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. ECCV 2024. arXiv:2303.05499.
[11] Ruan X. & Tang W. Fully Test-Time Adaptation for Object Detection. CVPR Workshops 2024, pp. 1038–1047.
[12] CIE 88:2004. Guide for Lighting of Road Tunnels and Underpasses. International Commission on Illumination.
[13] Yang et al. Why Does Driver Attention Abnormally Decrease? Transportation Research Part F, 2025.
[14] Onzon E., Bömer M., Mannan F., Heide F. Neural Exposure Fusion for High-Dynamic Range Object Detection. CVPR 2024.
[15] Alcantarilla P.F. et al. Night Time Vehicle Detection for Driving Assistance LightBeam Controller. IEEE Intelligent Vehicles Symposium (IV), 2008.
[16] Solovyev R. et al. Weighted Boxes Fusion: Ensembling Boxes from Different Object Detection Models. Image and Vision Computing, 2021.
