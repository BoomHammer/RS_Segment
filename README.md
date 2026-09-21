# RS-Segment

基于弱监督学习的遥感植被语义分割项目。主流程采用 PointSAM 生成伪标签、窗口化数据集切分，以及 AnySat 层级分割训练；所有大栅格均按窗口流式读取。

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

主流程以 `src/pipeline.py` 为编排入口，依次执行：

```text
scripts/preprocess.py  数据校验、流式统计、PointSAM 弱标签及质量报告
        ↓
scripts/datasets.py    窗口索引、多源对齐、空间划分
        ↓
scripts/train.py       AnySat 多模态共享编码器、层级监督训练
        ↓
scripts/test.py        空间测试集实测样点评估
        ↓
scripts/predict.py     halo 中心裁剪、重叠高斯融合、GeoTIFF 与统一类别表
```

训练保留 BF16、梯度累积和 halo 上下文，栅格按窗口读取，以适配单张
RTX 4090（24GB）。当前目标网格为 EPSG:4326、0.00225°，属于近似
250m 网格，并非各纬度严格等距的 250m；严格米制制图需另行配置投影网格。
调试诊断与像素 DNN 对照入口已清理；主流程回归测试保留在 `scripts/test/`。

| 文件或命令 | 作用 | 当前状态 |
| --- | --- | --- |
| `rs-prepare-data` / `scripts/prepare_data.py` | 无参数执行数据目录检查与栅格统计 | 可用 |
| `scripts/validate_labels.py` | 单独运行样点读取、校验和类别映射 | 可用 |
| `scripts/weak_label.py` | 使用 PointSAM 生成伪标签及质量报告 | 可用 |
| `scripts/datasets.py` | 根据伪标签 run 生成样本索引、空间划分、验证和基准报告 | 可用 |
| `scripts/train.py` | 使用准备好的窗口数据集训练 AnySat | 可用 |
| `scripts/stage2.py` | `datasets.py` 的历史兼容入口 | 兼容 |
| `scripts/predict.py` | 重叠滑窗全图推理和 GeoTIFF 输出 | 可用 |
| `scripts/test.py` | 在空间测试集上评估 checkpoint | 可用 |
| `rs-pipeline` / `src/pipeline.py` | 串联数据准备、训练、测试和全图预测 | 可用 |
| `rs-train-predict` | 训练完成后自动全图预测 | 可用 |
| `rs-resume-predict` | 从 `last.pt` 续训完成后自动全图预测 | 可用 |

## 命令行用法

### 1. 流程外的常用命令

#### 1.1 一键完成全流程

串联数据准备、弱标签、数据集构建、训练、测试和全图预测：

```bash
uv run rs-pipeline
```

如果数据集已经准备好，只需要训练并在训练结束后自动预测：

```bash
uv run rs-train-predict data/processed/<数据集时间戳>
```

训练中断后，从完整断点续训并自动预测：

```bash
uv run rs-resume-predict experiments/<训练实验时间戳>/last.pt
```

原有全流程命令也支持续训模式，会跳过数据准备和数据集构建：

```bash
uv run rs-pipeline --resume experiments/<训练实验时间戳>/last.pt
```

也可以使用 `uv run python -m pipeline`，并通过 `--epochs`、`--device` 及各配置参数覆盖默认设置。

#### 1.2 检查标签

单独检查样点读取、校验和类别映射：

```bash
uv run python scripts/validate_labels.py --config configs/data.yaml
```

### 2. 主流程

按以下顺序执行各阶段命令。`<数据集时间戳>` 和 `<训练实验时间戳>` 替换为实际生成的目录名。

#### 1.1 准备数据

检查数据目录、标签和配置，并计算或复用栅格统计结果：

```bash
uv run rs-prepare-data
```

#### 1.2 生成 PointSAM 弱标签

生成伪标签、标签映射和质量报告，输出到 `data/processed/<YYYYMMDD_HHMMSS>`：

```bash
uv run python scripts/weak_label.py
```

#### 1.3 构建窗口数据集

根据上一阶段的 run 生成样本索引、空间划分和验证报告：

```bash
uv run python scripts/datasets.py data/processed/<数据集时间戳>
```

#### 1.4 训练模型

使用准备好的窗口数据集训练 AnySat，产物写入 `experiments/<训练实验时间戳>`：

```bash
uv run python scripts/train.py data/processed/<数据集时间戳>
```

`configs/train.yaml` 的 `training.amp_dtype: auto` 会在 RTX 4090 上选择
BF16，在 RTX 2070 SUPER 上选择 FP16 并启用梯度缩放；CPU 使用 FP32。
训练启动时会打印实际精度，验证使用同一设置。旧配置指定 BF16 但显卡不支持
原生 BF16 时，也会自动回退到 FP16。AnySat 使用梯度检查点、SDPA 和分块计算，
默认 Tiny、`patch_size: 32`、`subpatch_size: 4`，各产品保留全部实际观测日期。
320×320 的含 halo 窗口在当前10个模态下共有1001个 combiner token；
`max_tokens: 2048` 防止超预算。增大模型或窗口前须重新检查显存。
训练启用重叠一致性损失，预测沿用 halo 中心裁剪与高斯融合。
验证使用独立的小型 worker 池（`validation_num_workers`），结束后退出，
不启用页锁定预取；返回训练前释放空闲 CUDA 缓存。训练进度显示 `data`（等待数据）、
`step`（传输及训练计算）和 `VRAM`（当前张量占用 / CUDA 缓存保留，GiB），
用于排查跨轮次速度下降。VRAM 数字不是显卡总占用，也不是单步峰值。

架构、数据适配、与论文的差异和真实窗口检查命令见
[AnySat 说明](docs/anysat.md)。默认 Tiny 随机初始化，需要重新训练；
可选 `configs/model_anysat_base.yaml` 加载官方 Base 预训练核心，新增数据投影器
独立学习。旧模型 checkpoint 仍按保存的架构读取，不能直接作为 AnySat 续训权重。

训练中断后可使用 `last.pt` 续训；续训必须使用原数据集和实验配置：

```bash
uv run python scripts/train.py \
  data/processed/<数据集时间戳> \
  --resume experiments/<训练实验时间戳>/last.pt
```

#### 1.5 测试模型

默认在空间测试集的实测样点上评估 checkpoint，也可通过 `--split validation` 测试验证集：

```bash
uv run python scripts/test.py \
  experiments/<训练实验时间戳>/model_<训练实验时间戳>.pt
```

#### 1.6 全图预测

使用 checkpoint 执行重叠滑窗推理并输出 GeoTIFF：

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

## 测试与代码检查

项目使用 pytest 和 Ruff：

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

测试文件当前位于 [`scripts/test`](scripts/test)；pytest 配置已在 `pyproject.toml` 中设置测试路径和 `src` 导入路径。

完整流程会在 `data/processed/<时间戳>/` 保存数据准备和数据集产物，在 `experiments/<时间戳>/` 保存 checkpoint、配置快照、训练日志、测试指标以及预测结果。预测使用重叠滑窗和高斯融合，默认按配置写出 GeoTIFF；所有阶段按窗口流式读取，不会把 100GB 级原始栅格一次性加载到内存。
