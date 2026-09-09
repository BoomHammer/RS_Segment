# RS-Segment

基于弱监督学习的遥感植被语义分割项目。主流程采用 PointSAM 生成伪标签、窗口化数据集切分，以及 SegFormer-U-TAE 训练；所有大栅格均按窗口流式读取。

## 环境准备

- Python `>=3.11,<3.14`
- CUDA `12.4`（使用 GPU 的模型功能启用后需要）
- 包管理器：`uv`

在项目根目录执行：

```bash
uv sync --extra dev
```

如果暂时不能使用 `uv`，也可以直接使用项目环境中的 Python、pytest 和 Ruff。命令示例中的 `uv run` 可替换为对应环境的 Python 调用。

## 配置文件

数据路径通过 [`configs/data.yaml`](configs/data.yaml) 配置，PointSAM 参数独立放在 [`configs/weak_label.yaml`](configs/weak_label.yaml)，模型和训练参数分别放在 `configs/model.yaml` 与 [`configs/train.yaml`](configs/train.yaml)，全图预测参数放在 [`configs/predict.yaml`](configs/predict.yaml)。

相对路径均相对于配置文件所在目录解析，而不是相对于当前 shell 目录解析。提交命令前，请根据本地数据位置调整配置。

## 程序入口

| 文件或命令 | 作用 | 当前状态 |
| --- | --- | --- |
| `rs-prepare-data` / `scripts/prepare_data.py` | 无参数执行数据目录检查与栅格统计 | 可用 |
| `scripts/validate_labels.py` | 单独运行样点读取、校验和类别映射 | 可用 |
| `data.filename_parser` | 解析动态影像文件名并生成元数据 | 可用 |
| `scripts/weak_label.py` | 使用 PointSAM 生成伪标签及质量报告 | 可用 |
| `scripts/datasets.py` | 根据伪标签 run 生成样本索引、空间划分、验证和基准报告 | 可用 |
| `scripts/train.py` | 使用准备好的窗口数据集训练 SegFormer-U-TAE | 可用 |
| `scripts/stage2.py` | `datasets.py` 的历史兼容入口 | 兼容 |
| `scripts/predict.py` | 重叠滑窗全图推理和 GeoTIFF 输出 | 可用 |
| `scripts/test.py` | 在空间测试集上评估 checkpoint | 可用 |
| `rs-pipeline` / `src/pipeline.py` | 串联数据准备、训练、测试和全图预测 | 可用 |

## 命令行用法

### 1. 数据准备

安装项目后推荐使用：

```bash
uv run rs-prepare-data
```

该命令固定读取 `configs/data.yaml`，先检查目录、标签文件和配置，再计算或复用栅格统计结果。

不安装命令行脚本时可使用：

```bash
uv run python scripts/prepare_data.py
```

### 1.1 一键完成全流程

推荐在项目根目录执行下面的命令。它会依次创建预处理 run、生成 PointSAM 弱标签、建立窗口索引和空间划分、训练模型、计算测试指标，并生成全图预测 GeoTIFF 及类别对照表：

```bash
uv run rs-pipeline
```

也可以不安装命令行脚本直接运行：

```bash
uv run python -m pipeline
```

常用覆盖参数如下：

```bash
uv run rs-pipeline --epochs 30 --device cuda
uv run rs-pipeline --data-config configs/data.yaml --model-config configs/model.yaml \
  --train-config configs/train.yaml --predict-config configs/predict.yaml
```

### 2. 独立生成 PointSAM 伪标签

伪标签生成是主流程的第一步，会自动创建 `data/processed/<YYYYMMDD_HHMMSS>`，并将标签映射、GeoTIFF 伪标签和质量报告写入其中：

```bash
uv run python scripts/weak_label.py
```

PointSAM 参数统一配置在 [`configs/weak_label.yaml`](configs/weak_label.yaml)，数据路径和标签字段配置在 [`configs/data.yaml`](configs/data.yaml)。

### 3. 主流程：准备数据集和训练

完成上一节的伪标签生成后，继续执行以下命令：

```bash
uv run python scripts/datasets.py data/processed/<YYYYMMDD_HHMMSS>
uv run python scripts/train.py data/processed/<YYYYMMDD_HHMMSS>
```

目录结构如下：

```text
data/processed/<YYYYMMDD_HHMMSS>/
├── label_mapping.json
├── weak_labels.tif
├── weak_labels_quality.json
├── weak_labels_quality.png
├── sample_index.json
├── spatial_split.json
├── stage2_validation.json
├── stage2_benchmark.json
└── raster_stats_stage2.json
```

训练完成后，关键训练产物写入：

```text
experiments/<YYYYMMDD_HHMMSS>/
├── model_<YYYYMMDD_HHMMSS>.pt
├── last.pt
├── best_loss.pt
├── best_accuracy.pt
├── best_unique_accuracy.pt
├── epoch_metrics.csv
├── train_log.json
├── train.yaml
├── model.yaml
├── data.yaml
└── test_metrics.json
```

每次训练均使用训练启动时间创建独立实验目录，因此同一数据集可以重复训练。`train_log.json` 合并保存训练元数据、每个 epoch 的 loss/accuracy、验证指标、学习率、最佳 epoch 和早停信息。

#### 断点续训

每轮验证完成后，训练入口会将完整训练状态原子写入实验目录中的 `last.pt`。中断后使用原数据集目录和该文件继续训练：

```bash
uv run python scripts/train.py \
  data/processed/<原数据集时间戳> \
  --resume experiments/<原训练实验时间戳>/last.pt
```

`last.pt` 包含当前模型、AdamW 优化器、学习率调度器、EMA、历史最佳权重、早停计数、随机数状态和最近完成的 epoch。续训从最近完整结束的下一轮开始；如果中断发生在某轮训练或验证过程中，该轮没有写入断点，需要重新执行。第一轮验证完成前不会生成可用的 `last.pt`。

续训会自动使用原实验目录中的 `train.yaml`、`model.yaml`、`data.yaml` 配置快照，以及断点保存的窗口参数和原定总 epoch 数。传入的数据集目录必须与断点记录完全一致，不能通过 `--epochs` 改变原定总轮数，也不能把续训输出写入另一个实验目录。

`model_*.pt`、`best_loss.pt`、`best_accuracy.pt` 和 `best_unique_accuracy.pt` 只保存推理权重，不能用于 `--resume`；只有完整断点 `last.pt` 可以恢复训练。多进程 DataLoader worker 内部的随机状态不会保存，因此续训后的结果不保证与从未中断的训练逐位完全一致。

恢复训练时，`epoch_metrics.csv` 会继续逐轮追加且不会重复写同一个 epoch。旧实验没有该 CSV 时，只能从恢复后的下一轮开始记录，无法补算之前各轮的训练真实标签和伪标签独立指标。

训练完成后，只需将 checkpoint 作为测试入口参数。程序会从同目录的 `train_log.json` 自动找到训练数据和配置：

```bash
uv run python scripts/test.py experiments/<训练实验时间戳>/model_<训练实验时间戳>.pt
```

测试默认使用 `spatial_split.json` 中的 `test` 划分，并只在实测样点上统计指标。
可用 `--split validation` 测试验证集，或用 `--max-windows` 限制窗口数量进行快速检查。
测试结果直接写入 checkpoint 所在的实验目录：

```text
experiments/<YYYYMMDD_HHMMSS>/
├── train_log.json     # 训练配置、来源和完整训练过程
├── test_metrics.json  # loss、Accuracy、Precision、Recall、F1、ROC/AUC、MSE、MAE、IoU
└── model_<YYYYMMDD_HHMMSS>.pt
```

训练入口可通过 `--data-config`、`--config`、`--train-config`、`--epochs` 和 `--device` 覆盖默认设置；测试入口只需提供 checkpoint，另支持 `--split`、`--max-windows` 和 `--device`。

如只需检查标签：

```bash
uv run python scripts/validate_labels.py --config configs/data.yaml
```

数据集阶段的特征清单、时间范围、缺帧策略、窗口大小和步长、空间块划分、采样、
DataLoader worker、缓存、BF16、梯度累积及基准批次数均在
`configs/data.yaml` 的 `data.stage2` 中配置，不需要重复写入命令行。

`data.stage2.split` 的 `ratios` 按 `[train, validation, test]` 顺序配置空间划分比例；
划分以空间块为不可拆分单元，并根据地面实测类别进行分层分配，减少验证集或测试集缺少类别的情况。
`data.stage2.sampling.split: train` 仅表示训练阶段的加权采样从 `train` 划分取样。
伪标签只进入训练损失，验证和测试指标只使用地面实测标签。

### 4. 全图预测

使用训练 checkpoint 执行重叠滑窗推理：

```bash
uv run python scripts/predict.py \
  experiments/<训练实验时间戳>/model_<训练实验时间戳>.pt \
  --config configs/predict.yaml
```

预测参数包括窗口大小、步长、高斯融合、设备、AMP 和输出压缩方式。默认情况下，结果图完全来自模型预测；
如果需要将所有地面实测标签写回结果图对应像素，在 `configs/predict.yaml` 中设置：

```yaml
predict:
  override_ground_truth: true
```

该覆盖只作用于输出 GeoTIFF，不会改变模型推理或训练数据。

### 5. 解析动态影像文件名

解析目录下顶层的 `.tif` 文件，并生成 JSON 或 YAML 元数据：

```bash
uv run python -m data.filename_parser \
  data/raw/dynamic \
  --output experiments/dynamic_metadata.json
```

输出 YAML：

```bash
uv run python -m data.filename_parser \
  data/raw/dynamic \
  --output experiments/dynamic_metadata.yaml
```

支持的文件名格式：

| 格式 | 示例 | 解析结果 |
| --- | --- | --- |
| `CATEGORYyymmdd.tif` | `GPP230218.tif` | 日尺度，日期为 `2023-02-18` |
| `CATEGORYyymmddB<n>.tif` | `SR230805B6.tif` | 日尺度，并记录波段 `6` |
| `CATEGORYyymm.tif` | `SOIL2004.tif` | 月尺度，月份为 `2020-04` |

类别会统一转为大写。命令只扫描指定目录的顶层 `.tif` 文件；文件名无法匹配时会报错。输出文件的后缀必须是 `.json`、`.yaml` 或 `.yml`，且输出父目录需要预先存在。

## 测试与代码检查

项目使用 pytest 和 Ruff：

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

测试文件当前位于 [`scripts/test`](scripts/test)；pytest 配置已在 `pyproject.toml` 中设置测试路径和 `src` 导入路径。

完整流程会在 `data/processed/<时间戳>/` 保存数据准备和数据集产物，在 `experiments/<时间戳>/` 保存 checkpoint、配置快照、训练日志、测试指标以及预测结果。预测使用重叠滑窗和高斯融合，默认按配置写出 GeoTIFF；所有阶段按窗口流式读取，不会把 100GB 级原始栅格一次性加载到内存。
