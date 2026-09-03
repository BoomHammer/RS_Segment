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

数据路径通过 [`configs/data.yaml`](configs/data.yaml) 配置，PointSAM 参数独立放在 [`configs/weak_label.yaml`](configs/weak_label.yaml)，模型和训练参数分别放在 `configs/model.yaml` 与 `configs/train.yaml`。

```yaml
data:
  root: ../data
  labels: ../data/labels
  raw: ../data/raw
  dynamic: ../data/raw/dynamic
  static: ../data/raw/static
  processed: ../data/processed
  label_file: ../data/labels/traindata20250626ori.csv
  output_nodata: -9999
  label_crs: EPSG:4326
  label_schema:
    columns:
      index: Index
      x: X
      y: Y
      formation: Eng_Formation
      alliance: Eng_Alliance
      chn_formation: Formation
      chn_alliance: Alliance
  target_grid:
    crs: EPSG:4326
    resolution: [0.00225, 0.00225]
  raster:
    nodata: -9999
```

相对路径均相对于配置文件所在目录解析，而不是相对于当前 shell 目录解析。提交命令前，请根据本地数据位置调整配置。

标签契约中 `formation` 是英文大类字段，`alliance` 是英文小类字段，分别对应
`Eng_Formation` 和 `Eng_Alliance`；中文字段只用于输出映射。`Index` 保留为样点自身编号，类别编码使用 `formation_code` 和 `alliance_code`。`output_nodata` 是输出结果的无效值，默认为 `-9999`。

`target_grid` 规定所有空间对齐和样点定位使用的目标 CRS 与分辨率。默认是 WGS84、
`0.00225°`；如需使用 Albers 等面积投影，可改为对应 CRS 和 `250` 米分辨率。

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
| `scripts/predict.py` | 推理入口 | 预留，尚未实现 |
| `scripts/test.py` | 在空间测试集上评估 checkpoint | 可用 |

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
├── train_log.json
├── train.yaml
├── model.yaml
├── data.yaml
└── test_metrics.json
```

每次训练均使用训练启动时间创建独立实验目录，因此同一数据集可以重复训练。`train_log.json` 合并保存训练元数据、每个 epoch 的 loss/accuracy、验证指标、学习率、最佳 epoch 和早停信息。

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

### 4. 解析动态影像文件名

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



PointSAM 弱监督标签生成、SegFormer-U-TAE 训练和数据集切分已接入；带重叠滑窗及高斯加权的全图推理将在后续阶段接入。
