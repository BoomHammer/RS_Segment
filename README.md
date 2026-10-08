# RS-Segment

面向大陆级遥感植被制图的弱监督语义分割项目。结合稀疏实测样点、多时相动态影像和静态环境特征，支持 PointSAM 伪标签、层级分类及多种分割模型，目标是生成 250 m 分辨率的植被类型分布图。

数据按窗口流式读取；训练使用重叠区域一致性约束，预测使用带 halo 的重叠滑窗和高斯融合，以减少拼接边界伪影。默认模型为轻量双分支模型（`lightweight_dual_branch`），模型切换与对照实验参见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 环境与安装

- Python `>=3.11,<3.14`
- 包管理器：uv
- GPU 环境：PyTorch CUDA 12.4；主要面向单张 RTX 3090，自动切换多卡运行
- 代码格式化与检查：Ruff

在项目根目录安装依赖：

```powershell
uv sync --extra dev
```

生成 PointSAM 伪标签还需要可导入的 `sam2` 包和本地 SAM2 权重；`uv sync` 不包含 SAM2 安装或权重下载。权重路径由 `configs/data.yaml` 中的 `data.sam2_checkpoint` 指定。复用已有数据集训练时不调用 SAM2。

## 数据与配置

```text
configs/               # 数据、模型、训练和预测配置
data/
  labels/              # 实测样点 CSV
  raw/dynamic/         # 多时相影像
  raw/static/          # DEM、气候等静态影像
  processed/           # 预处理产物、样本索引及空间划分
experiments/           # 训练断点、指标、配置快照和预测结果
scripts/               # 各阶段入口；test/ 下为自动化测试
src/                   # 数据、模型、损失、推理等实现
```

首次运行前，根据本机数据修改以下配置：

| 配置 | 主要内容 |
| --- | --- |
| [data.yaml](configs/data.yaml) | 数据路径、CSV 编码与坐标系、标签层级、目标网格、窗口与空间划分 |
| [weak_label.yaml](configs/weak_label.yaml) | PointSAM 输入组合、置信度与融合策略 |
| [model.yaml](configs/model.yaml) | 模型架构、监督权重与模型参数 |
| [train.yaml](configs/train.yaml) | 训练轮数、采样、混合精度、梯度累积与早停 |
| [predict.yaml](configs/predict.yaml) | 推理设备、融合方式、输出与实测标签覆盖策略 |

`data.yaml` 中的数据路径相对于该配置文件所在目录解析；命令行路径相对于当前工作目录。样点 CSV 的列名、编码、坐标系与分类层级必须与配置一致，标签契约详见 [docs/label_hierarchy.md](docs/label_hierarchy.md)。

## 运行流程

以下命令均在项目根目录执行，尖括号内容需替换为实际目录名。

### 从原始数据开始

```powershell
uv run rs-pipeline
```

依次执行：预处理与栅格统计 → PointSAM 伪标签生成 → 样本索引与空间划分 → 训练 → 测试 → 全图预测。

### 伪标签生成断点续跑

PointSAM 生成中断后，使用原预处理目录恢复，跳过已完成的样点：

```powershell
uv run python scripts/weak_label.py --data-config configs/data.yaml --config configs/weak_label.yaml --resume data/processed/<数据集时间戳>
```

恢复时需保持目标网格、输入影像、标签和弱标签配置不变；如果原任务使用自定义配置路径，继续使用原路径。多卡任务还需保持与中断前相同的可见 GPU 数量。此命令只恢复伪标签生成，完成后可继续执行上述数据集构建、训练、测试和预测步骤。

### 复用已有数据集

```powershell
uv run rs-pipeline --retrain data/processed/<数据集时间戳>
```

使用已有样本索引、标签映射、统计量和空间划分，新建实验并从头训练，随后测试和全图预测。不会重新预处理、生成伪标签或划分数据集；索引引用的影像与标签文件仍需可访问。

仅用真实标签、合并测试集训练，以及 SegFormer+U-TAE、U-TAE、SegFormer、MAESTRO-S、AnySat 模型切换，参见 [EXPERIMENTS.md](EXPERIMENTS.md)。

### 从断点续训

```powershell
uv run rs-pipeline --resume experiments/<实验时间戳>/last.pt
```

恢复原实验的配置、数据集路径及完整训练状态，完成后测试和全图预测；原实验合并了测试集时自动跳过测试。续训只接受 `last.pt`，不能使用 `model_*.pt` 或 `best_*.pt`。`--resume` 与 `--retrain` 不能同时使用。

### 常用参数

```powershell
uv run rs-pipeline --retrain data/processed/<数据集时间戳> --epochs 30 --device cuda:0
uv run rs-pipeline --data-config configs/data.yaml --model-config configs/model.yaml --train-config configs/train.yaml --predict-config configs/predict.yaml
uv run rs-pipeline --help
```

`--epochs` 用于设置新训练的目标轮数，续训须保持原目标轮数。检测到多张可见 GPU 时，相关阶段支持自动多卡运行；显式指定 `--device cuda:0` 可选择单卡。训练耗时分析与性能诊断参见 [PERFORMANCE.md](PERFORMANCE.md)。

### 分步执行

需要单独运行某个阶段时：

```powershell
uv run python scripts/preprocess.py --config configs/data.yaml --skip-weak-labels
uv run python scripts/weak_label.py --data-config configs/data.yaml --run-dir data/processed/<数据集时间戳>
uv run python scripts/datasets.py data/processed/<数据集时间戳>
uv run python scripts/train.py data/processed/<数据集时间戳>
uv run python scripts/test.py experiments/<实验时间戳>/model_<实验时间戳>.pt
uv run python scripts/predict.py experiments/<实验时间戳>/model_<实验时间戳>.pt --config configs/predict.yaml
```

各脚本的其他选项通过 `--help` 查看。

## 输出与预测

中间数据保存在 `data/processed/<数据集时间戳>/`。实验目录通常包含：

| 产物 | 用途 |
| --- | --- |
| `last.pt` | 可续训的完整状态 |
| `model_<实验时间戳>.pt`、`best_*.pt` | 测试与预测权重 |
| `train_log.json` | 训练记录、来源数据集与监督策略 |
| `data.yaml`、`model.yaml`、`train.yaml` | 生效配置快照 |
| `spatial_split.json`、`supervision_audit.json` | 本次实验的空间划分视图与监督审计 |
| `test_metrics.json` | 测试指标；跳过测试时不生成 |
| `vegetation_*.tif`、`vegetation_*.csv` | 植被分类 GeoTIFF 与类别映射表 |

预测沿用训练记录中的窗口参数。当前 `configs/predict.yaml` 默认设置 `override_ground_truth: true`，会用实测标签覆盖输出图中的对应像元；如需保留纯模型预测，设为 `false`。该选项不改变训练或测试指标。默认禁止覆盖同名预测文件，重复出图需指定新输出路径或明确启用 `overwrite`。

## 开发检查

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

MAESTRO-S 与 AnySat 的实现和权重说明分别参见 [docs/maestro.md](docs/maestro.md) 和 [docs/anysat.md](docs/anysat.md)。
