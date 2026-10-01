# 训练手册（B0~B3 + 改进算法接入）

> 给上云训练的同学（T1）——从零到消融表全流程。
> 所有命令都在仓库根目录执行；数据下载链路已于 2026-10-01 在国内网络实测打通。

---

## 0. 环境（本地 N 卡 或 AutoDL，二选一）

### 0a. 本地 NVIDIA 显卡（队内有 RTX 5060 ✅，零费用）

⚠️ **50 系卡（Blackwell 架构）必读**：必须装 CUDA 12.8 版 PyTorch（2.7 以上），
旧版 torch 会报 `no kernel image is available for execution on the device`。

```bash
git clone https://github.com/kyleliui121/aic_project.git && cd aic_project
pip install -r requirements.txt
# 关键一步：重装支持 50 系的 torch（覆盖 ultralytics 带的默认版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 训练前 30 秒自检（三个都要对）：
python -c "import torch; print('torch', torch.__version__, '| cuda可用:', torch.cuda.is_available(), '| 显卡:', torch.cuda.get_device_name(0))"
python -c "import torch; x=torch.randn(8,8).cuda(); print('GPU计算测试:', (x@x).sum().item() != 0)"
```

- 5060（8GB 显存）跑 yolo11n：batch 16、imgsz 640，显存占用约 3~4GB，**余量充足**
- 速度预期：B0 一次训练约 40~60 分钟；显存富余可开 batch=32 提速
- 显卡空闲时也可顺手跑 yolo11s 加一行消融（奖励项）

### 0b. AutoDL 备用（仅当本地卡不能用）

租 RTX 3090（数据盘 50GB+），镜像选 **PyTorch 2.x + CUDA 12.x + Python 3.10**，
费用预估见 §5。以下命令两套环境**一字不改通用**。

## 1. 数据下载（国内直连，已实测 ✅）

BDD100K 用 ModelScope 阿里官方镜像（70k 训练 / 10k 验证，含检测标注 json）：

```bash
python -c "
from modelscope.msdatasets import MsDataset
import warnings; warnings.filterwarnings('ignore')
MsDataset.load('iic/BDD100K', split='train')        # train.zip：7万图 + bdd100k_labels_images_train.json
MsDataset.load('iic/BDD100K', split='validation')   # val.zip：1万图 + bdd100k_labels_images_val.json
"
```

- 下载+解压到 `~/.cache/modelscope/hub/datasets/iic/BDD100K/master/data_files/extracted/<hash>/`
- 实测速度：val.zip（约780MB）约 2 分钟；train.zip（约6GB）预计 15~30 分钟
- 已验证内容：val 标注 json 208MB 可解析，10000 图 / 279,237 框，其中**夜晚子集 3929 图 / 98,940 框**

EXDark（低照度测试场③）在 HuggingFace 国内镜像，已是 YOLO 格式且分好折：

```bash
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(repo_id='dronefreak/ExDark', repo_type='dataset', local_dir='datasets/exdark_raw')
"
# data/train|valid|test 各含 images/ + labels/（YOLO txt），直接可用
```

## 2. 转 YOLO 格式 + 昼夜切分

```bash
EXT=$(find ~/.cache/modelscope -path "*extracted*" -name "bdd100k_labels_images_train.json" | head -1 | xargs dirname)
python data_prep/bdd_to_yolo.py \
    --labels "$EXT/bdd100k_labels_images_train.json" \
    --images "$(dirname $(dirname "$EXT"))/images" \
    --out datasets
```

产出 `datasets/bdd_day.yaml / bdd_night.yaml / bdd_dusk.yaml`（自动按 timeofday 切分、
各自独立 train/val 划分防数据污染）+ `split_report.md` 统计。
⚠️ 此脚本未在真实 BDD json 上跑过，首次运行报错发回仓库 issue。

## 3. 训练矩阵（核心消融）

```bash
# B0 基线：白天子集微调（所有对比的起点）
yolo train model=yolo11n.pt data=datasets/bdd_day.yaml epochs=50 imgsz=640 batch=16 name=B0

# B1 光照增强：在线 HSV 亮度抖动拉满（ultralytics 内置参数）
yolo train model=yolo11n.pt data=datasets/bdd_day.yaml epochs=50 imgsz=640 batch=16 \
      hsv_v=0.9 hsv_s=0.5 name=B1

# B2 真夜晚：白天+夜晚混合（生成合并 yaml）
python - <<'EOF'
open('datasets/bdd_mix.yaml','w').write(
"""path: {ABS}/datasets
train: [bdd_day/images/train, bdd_night/images/train]
val: bdd_day/images/val
nc: 10
names: [person, car, bus, truck, bike, motor, rider, light, sign, train]
""".replace('{ABS}','<你的绝对路径>'))
EOF
yolo train model=yolo11n.pt data=datasets/bdd_mix.yaml epochs=50 imgsz=640 batch=16 name=B2

# B3 = B1 参数 + bdd_mix.yaml 数据

# ALT（光照对抗，可选加分项）：见 §4.2
```

## 4. 改进算法接入（本仓库算法怎么嵌进流水线）

### 4.1 全景图

```
bdd_to_yolo.py ──► 训练集 ──► B0/B1/B2/B3 训练 ──► best.pt（每版各存一份）
                                     │
train_alt.py（用B0权重搜最坏光照）────┘（产出alt_data并回炉= B1+）
train_lapp.py（冻结B0，训提亮小网络）──► lapp.pt（推理端预处理=C3/C4）
                                              │
eval_matrix.py ◄── 全部best.pt + 各测试场 ──► matrix.md（报告表格）
video_detect.py / make_demo_data.py ◄─ best.pt ──► 演示网页/答辩视频
warning.py ──► 预警输出（第④层）
```

### 4.2 ALT 光照对抗训练（B1 的升级）

```bash
# 阶段1：用 B0 权重在白天训练集上搜"最坏光照"并导出
python experiments/train_alt.py \
    --data datasets/bdd_day/images/train --labels datasets/bdd_day/labels/train \
    --model runs/detect/B0/weights/best.pt --steps 3 --out datasets/bdd_day_alt

# 阶段2：原始图 + 最坏光照图合并（文件名不冲突直接cp），标准训练
cp datasets/bdd_day_alt/images/* datasets/bdd_day/images/train/
cp datasets/bdd_day_alt/labels/* datasets/bdd_day/labels/train/
yolo train model=yolo11n.pt data=datasets/bdd_day.yaml epochs=50 imgsz=640 name=B1plus
```

### 4.3 LA-PP 自适应预处理（C3/C4）

```bash
# 冻结 B0 权重，训提亮小网络（GPU 约1~2小时）
python experiments/train_lapp.py \
    --data datasets/bdd_day/images/train --labels datasets/bdd_day/labels/train \
    --model runs/detect/B0/weights/best.pt --epochs 3 --bs 8 --out lapp_weights.pt

# 评测时：测试图先过 lapp（--apply 模式）再送任意检测器
python experiments/train_lapp.py --apply lapp_weights.pt --image 测试图.jpg --out 增强图.jpg
```

### 4.4 统一评测出表

```bash
python experiments/eval_matrix.py \
    --models B0=runs/detect/B0/weights/best.pt B1=runs/detect/B1/weights/best.pt \
             B2=runs/detect/B2/weights/best.pt B3=runs/detect/B3/weights/best.pt \
    --sets 白天val=datasets/bdd_day.yaml 夜晚val=datasets/bdd_night.yaml \
           EXDark=datasets/exdark.yaml \
    --out results/matrix.md
# 受控退化测试集（L0~L4）另用 degrade_light.py 生成后同样喂进来
```

### 4.5 结果回灌演示网页

```bash
python demo/scripts/make_demo_data.py   # 用真实检测结果替换占位框（内含 MEF 管线）
```

## 5. 费用与时长预估

**本地 RTX 5060 方案（当前采用）：费用 0 元**，B0~B3 + ALT + LA-PP + 评测约 10~12 小时（可分多次跑，训练结果自动保存在 runs/ 下）。

AutoDL 备用方案（3090）如需启用：

| 步骤 | 时长 | 费用 |
|---|---|---|
| B0~B3 四次训练 | 4×1h | ≈6元 |
| ALT 搜索+重训 | 2~3h | ≈4元 |
| LA-PP | 1~2h | ≈3元 |
| 评测矩阵 | 1h | ≈1.5元 |
| **合计** | **~10 GPU时** | **≈15元** |
