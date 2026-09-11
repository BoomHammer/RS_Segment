# RS-Segment

基于弱监督学习的遥感植被语义分割项目。数据按窗口流式读取，使用 PointSAM 生成弱标签，使用 SegFormer-U-TAE 训练，并通过重叠滑窗生成无缝 GeoTIFF。

## 环境准备

- Python `>=3.11,<3.14`
- CUDA `12.4`（GPU 训练/推理）
- [uv](https://docs.astral.sh/uv/)

```bash
uv sync --extra dev
```

首次运行前，按本机数据位置修改：

- [`configs/data.yaml`](configs/data.yaml)：数据、标签和预处理路径
- [`configs/weak_label.yaml`](configs/weak_label.yaml)：PointSAM
- [`configs/model.yaml`](configs/model.yaml)：模型
- [`configs/train.yaml`](configs/train.yaml)：训练
- [`configs/predict.yaml`](configs/predict.yaml)：全图预测

配置中的相对路径相对于配置文件所在目录解析。

## 命令行用法

### 1. 主流程

#### 1.1 新训练：训练、测试并预测

```bash
uv run rs-pipeline
```

该命令依次执行：数据预处理和弱标签生成 → 数据集索引与空间划分 → 训练 → 测试 → 全图预测。

常用覆盖参数：

```bash
uv run rs-pipeline --epochs 30 --device cuda
uv run rs-pipeline --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --predict-config configs/predict.yaml
```

#### 1.2 续训：从 `last.pt` 继续训练、测试并预测

```bash
uv run rs-pipeline --resume experiments/<训练实验时间戳>/last.pt
```

续训会自动读取断点所在实验目录的配置和数据集路径，完成后更新该实验的推理 checkpoint，并继续执行测试和全图预测。`--resume` 只接受完整断点 `last.pt`，不能使用 `model_*.pt` 或 `best_*.pt`。

如需指定设备：

```bash
uv run rs-pipeline --resume experiments/<训练实验时间戳>/last.pt --device cuda
```

#### 1.3 重新训练：复用已有数据集，从头训练并预测

```bash
uv run rs-pipeline --retrain data/processed/<数据集时间戳>
```

该命令跳过数据预处理和数据集重建，使用指定的已准备数据集新建实验，从头训练，随后自动测试和全图预测。可搭配 `--epochs`、`--device` 以及配置覆盖参数使用。

训练产物位于 `experiments/<时间戳>/`，主要包括：

```text
model_<时间戳>.pt       # 训练完成后用于测试/预测
last.pt                 # 可续训的完整状态
best_*.pt               # 各验证指标对应的推理权重
train_log.json          # 训练记录和来源数据集
test_metrics.json       # 测试指标
vegetation_*.tif        # 全图预测结果
```

#### 1.4 分步执行主流程

需要单独控制各阶段时，按以下顺序执行：

```bash
# 数据预处理、栅格统计和 PointSAM 弱标签
uv run python scripts/preprocess.py --config configs/data.yaml

# 使用上一步生成的目录建立样本索引和空间划分
uv run python scripts/datasets.py data/processed/<时间戳>

# 训练
uv run python scripts/train.py data/processed/<时间戳>

# 测试
uv run python scripts/test.py \
  experiments/<训练实验时间戳>/model_<训练实验时间戳>.pt

# 全图预测
uv run python scripts/predict.py \
  experiments/<训练实验时间戳>/model_<训练实验时间戳>.pt \
  --config configs/predict.yaml
```

`train.py` 单独续训时使用：

```bash
uv run python scripts/train.py data/processed/<时间戳> \
  --resume experiments/<训练实验时间戳>/last.pt
```

### 2. 流程之外的重要命令

检查数据目录并计算统计量：

```bash
uv run rs-prepare-data
```

检查样点标签：

```bash
uv run python scripts/validate_labels.py --config configs/data.yaml
```

解析动态影像文件名：

```bash
uv run python -m data.filename_parser data/raw/dynamic \
  --output experiments/dynamic_metadata.json
```

快速限制测试窗口数量：

```bash
uv run python scripts/test.py \
  experiments/<训练实验时间戳>/model_<训练实验时间戳>.pt \
  --split validation --max-windows 10 --device cuda
```

## 预测设置

预测默认使用训练 checkpoint 记录的窗口参数，并通过重叠滑窗和高斯融合输出 GeoTIFF。若需把实测标签覆盖回输出图，在 [`configs/predict.yaml`](configs/predict.yaml) 中设置：

```yaml
predict:
  override_ground_truth: true
```

该设置只影响输出，不改变训练或模型推理。

## 检查代码

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

完整流程的中间数据保存在 `data/processed/`，实验结果保存在 `experiments/`。所有大栅格按窗口流式读取，避免一次性加载到内存。
