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
# 数据预处理和栅格统计
uv run python scripts/preprocess.py --config configs/data.yaml --skip-weak-labels

# PointSAM 弱标签（已有多张 CUDA 卡时自动并行）
uv run python scripts/weak_label.py --data-config configs/data.yaml \
  --run-dir data/processed/<时间戳>

# PointSAM 中断后，从同一预处理目录的断点继续
uv run python scripts/weak_label.py --data-config configs/data.yaml \
  --resume data/processed/<时间戳>

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

PointSAM 会在输出目录中按样点保存临时进度；使用 `--resume` 后会跳过已经完成的样点，
而不是重新生成全部伪标签。恢复时必须继续使用原来的预处理目录，并保持目标网格、输入
影像、标签和弱标签配置不变，否则程序会拒绝加载不匹配的断点。多卡任务还应保持与中断
前相同的可见 GPU 数量。生成成功后，临时断点会自动清理。

如果中断的是包含弱标签生成的 `preprocess.py`，也可以直接恢复：

```bash
uv run python scripts/preprocess.py --config configs/data.yaml \
  --resume data/processed/<时间戳>
```

#### 1.5 单卡与多卡

`weak_label.py`、`train.py`、`test.py` 和 `predict.py` 会检查可见 CUDA
设备。普通命令检测到多张卡时会自动以每卡一个进程启动；单卡和 CPU 环境保持原有
行为。`rs-pipeline` 调用的也是这些入口，因此主流程同样自动适配多卡。

```bash
# 自动使用所有可见 GPU
uv run python scripts/train.py data/processed/<时间戳>

# 只开放两张指定 GPU
CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/train.py data/processed/<时间戳>

# 强制单卡（显式设备会关闭自动多卡）
uv run python scripts/train.py data/processed/<时间戳> --device cuda:0
```

也支持直接用 `torchrun` 启动。训练配置中的 `batch_size` 是每张卡的批量大小；
全局有效批量为 `batch_size × GPU 数 × gradient_accumulation_steps`。测试按窗口
分片并在全局像元坐标上去重；全图推理分别累加分数和高斯权重后流式归并，保持重叠
融合和无缝输出；PointSAM 分片同时保存置信分数，再按冲突阈值合并。

### 2. 流程之外的重要命令

若多卡 SAM 已完成、最终 `weak_labels.tif` 已写出，但质量报告汇总失败，且各卡的
`.weak_labels.tif.rankN`、`.scores.tif` 和 `.outcomes.json` 仍在，可核验整幅栅格并
恢复报告，无需再次运行 SAM：

```bash
uv run python -m scripts.recover_weak_labels data/processed/<时间戳> \
  --ranks 2 --conflict-margin 0.05
```

参数必须与原生成任务一致。此命令保留原栅格与分片，仅在逐块核验通过后写入质量报告
和预览图。随后运行 `scripts/datasets.py` 建立索引和空间划分，再使用
`rs-pipeline --retrain data/processed/<时间戳>` 继续训练、测试和出图；不要重新启动
无参数的 `rs-pipeline`，那会新建一次数据预处理与 SAM 任务。

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
