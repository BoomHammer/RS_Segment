# RS-Segment

基于弱监督学习的遥感植被语义分割项目。当前代码处于数据准备阶段，已提供数据目录检查、GeoTIFF 流式统计和动态影像文件名解析命令；训练与推理入口暂未实现。

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

数据相关命令通过 YAML 配置文件读取路径。默认配置为 [`configs/dataset.yaml`](configs/dataset.yaml)，主要结构如下：

```yaml
data:
  root: ../data
  labels: ../data/labels
  raw: ../data/raw
  dynamic: ../data/raw/dynamic
  static: ../data/raw/static
  processed: ../data/processed
  label_file: ../data/labels/traindata20250626ori.csv
  raster:
    nodata: -9999
```

相对路径均相对于配置文件所在目录解析，而不是相对于当前 shell 目录解析。提交命令前，请根据本地数据位置调整配置。

## 程序入口

| 文件或命令 | 作用 | 当前状态 |
| --- | --- | --- |
| `rs-check-data` / `scripts/check_data.py` | 检查数据目录、必需子目录和标签文件 | 可用 |
| `rs-compute-stats` / `scripts/compute_stats.py` | 按窗口流式计算 GeoTIFF 统计量 | 可用 |
| `data.filename_parser` | 解析动态影像文件名并生成元数据 | 可用 |
| `scripts/train.py` | 训练入口 | 预留，尚未实现 |
| `scripts/predict.py` | 推理入口 | 预留，尚未实现 |
| `scripts/test.py` | 测试入口 | 预留；当前使用 pytest |

## 命令行用法

### 1. 检查数据目录

安装项目后推荐使用：

```bash
uv run rs-check-data --config configs/dataset.yaml
```

不安装命令行脚本时可使用：

```bash
uv run python scripts/check_data.py --config configs/dataset.yaml
```

命令会检查配置中的数据根目录、`required_subdirectories` 指定的目录、`labels`、`raw` 以及可选的 `label_file`。

返回码：

- `0`：检查通过
- `1`：数据目录或标签文件缺失
- `2`：配置无法读取或格式无效

### 2. 计算 GeoTIFF 流式统计量

命令默认读取 `configs/dataset.yaml`，一次扫描配置中的 `data/raw/dynamic` 和
`data/raw/static` 两个目录，对其中全部顶层 `.tif` 影像计算统计量，并将结果缓存到
`data/processed`：

```bash
uv run rs-compute-stats
```

默认输出文件名类似 `raster_stats_20260831_163000.json`。也可以通过 `--output` 指定
输出位置，但建议仍放在 `data/processed`：

```bash
uv run rs-compute-stats \
  --output data/processed/raster_stats_manual.json \
  --band 1 \
  --window-size 512 512 \
  --nodata -9999
```

可用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--config` | `configs/dataset.yaml` | YAML 配置文件 |
| `--output` | 自动生成 | 输出 JSON 路径；默认写入 `data.processed` 并添加时间戳 |
| `--band` | `1` | 要统计的波段编号 |
| `--window-size WIDTH HEIGHT` | `1024 1024` | 分块读取窗口大小 |
| `--nodata` | 配置值 | 覆盖 `data.raster.nodata` |

统计过程按窗口读取影像，不会把整幅大图一次性载入内存；`NaN`、正负无穷和 NoData 像元会被排除。输出按类别组织：普通动态影像按文件名中的系列（如 `LST`）分组，带波段的 `SR230101B1.tif`、`SR230102B1.tif` 等按 `SR_B1` 分组，静态影像按文件名主干（如 `DSM`）分组。每个类别包含独立的 `statistics` 和 `files` 明细，动态与静态类别统一列在 `groups` 数组中。

命令会先检查 `data/processed/raster_stats_*.json` 中是否存在匹配缓存。只要输入影像的路径、文件大小、修改时间以及统计参数均未变化，就直接复用已有 JSON，不重复计算。

实际计算时使用单行进度条显示整体处理进度，不会为每个影像单独打印日志。

示例输出结构：

```json
{
  "schema_version": 3,
  "band": 1,
  "window_size": [1024, 1024],
  "nodata": -9999,
  "groups": [
    {
      "category": "SR_B1",
      "statistics": {
        "count": 123456,
        "mean": 0.42,
        "variance": 0.03,
        "standard_deviation": 0.17
      },
      "files": [
        {
          "path": "data/raw/dynamic/SR230101B1.tif",
          "date": "2023-01-01",
          "count": 123456,
          "mean": 0.42,
          "variance": 0.03,
          "standard_deviation": 0.17
        }
      ]
    }
  ]
}
```

动态文件明细会带有 `date`（日尺度）或 `month`（月尺度）字段；静态文件明细不包含时间字段。处理过程中，进度条末尾会显示当前正在处理的文件名。

### 3. 解析动态影像文件名

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

## 数据目录约定

```text
data/
├── labels/                 # 标签 CSV
├── raw/
│   ├── dynamic/             # 动态遥感影像
│   └── static/              # 静态遥感影像
└── processed/               # 预处理数据
```

训练、PointSAM 弱监督标签生成、Swin-U-TAE 模型和带重叠滑窗及高斯加权的全图推理将在后续阶段接入。
