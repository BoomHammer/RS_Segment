# AnySat 核心与现有数据适配

默认模型是 `configs/model.yaml` 的 `architecture: anysat`。
训练、测试、预测入口不变，模型输出仍为层级类别概率与有效像元掩码。
旧 MAESTRO / SegFormer-U-TAE checkpoint 按自身 contract 恢复，不能作为
AnySat 的续训断点。

## 实现来源

依据 [AnySat 论文](https://arxiv.org/html/2412.14123v3) 的 §3.1、§3.3，
采用传感器投影器、共享空间编码器、多模态 combiner 与 dense 分割头。
共享核心来自 [官方实现](https://github.com/gastruc/AnySat)，固定到
`5f6f475e1a22ce5e3a56a5b18f4ed6d24eca2a4a`，相关代码与 MIT 许可证
保存在 `src/models/anysat_vendor/`，运行无需克隆仓库或联网。

保留官方 Block、BlockTransformer、CrossBlockMulti 的参数结构与 iRPE。
`anysat_core.py` 使用 PyTorch SDPA、有效键掩码和梯度检查点执行这些层，
无需安装 Windows 下的 `flash-attn` / `rpe_ops` 扩展。
测试核对了无缺测时与官方 self-attention、iRPE 和完整 combiner 的数值一致性。

这是 AnySat 下游分割适配，没有实现或运行 GeoPlex JEPA 自监督预训练。
默认 Tiny（256维、4头、空间编码与融合各2层）随机初始化。
不同于直接调用官方 Hub：本项目新增了数据投影器与层级解码器，
共享核心可以加载官方 Base 权重；不假定新增投影器具有预训练能力。

## 数据对应关系

| 原有数据 | 接入方式 |
| --- | --- |
| `SR_B1`…`SR_B7` | 根据特征名绑定到 SR 多光谱模态，同一天的波段一起投影 |
| NDVI、EVI、FPAR、GPP、LAI 等 | 按产品拆分，各自使用实际观测日期与时间注意力 |
| 月温度、降水等 | 独立动态模态；月数据日期沿用数据集的月初约定 |
| DEM、地形、年均气候等静态特征 | 独立静态投影器，无伪造时间序列 |
| NaN、Inf、缺波段、补齐时间位置 | 结合逐值有限性和数据集掩码排除；有效性通道区分真实零值 |
| 稀疏实测点、PointSAM 伪标签 | 沿用原监督、空间划分、类别采样和两种标签的不同权重 |

每个动态产品保留全部已有时间位置，仅去除整个批次对该产品都没有观测的日期，
不做 MAESTRO 时间分箱，也不把时间轴当作输入通道。日期为零起始 DOY，支持
闰年；与官方季节编码一致，跨年的同一 DOY 不单独编码年份。
输入继续使用数据集已有有效范围、物理 Scale 和统计量标准化，模型不重复归一化。

自动推导的 `modalities` 必须恰好覆盖 contract 内的每个输入特征一次；支持通过
`[{name, role, features}]` 显式配置。通道按名称匹配，批次内不同顺序会报错。
`dense_modality: SR` 保留 SR 的空间子块特征；更改输入特征集合时，应同步选取
实际存在的 dense 模态。

现有流式读取器已经将所有栅格对齐到共同目标网格。因此 DEM 也按这个网格
进入模型，不能再标记为原生 90m。`patch_size`、`subpatch_size` 在本项目中
以目标网格像元计量，与官方 Hub 的米制 `patch_size` 不同。
官方 MODIS 特殊路径把 MODIS 作为整幅上下文 token；这里将 SR 作为新传感器
投影到空间 patch，保留区域内部差异以支持逐像元分割。

当前 EPSG:4326、0.00225° 网格的 `resolution_m: 250` 是明确的名义尺度，
不是各纬度严格等距的 250m。若需严格米制输出，应重建投影网格的数据集。
投影网格的真实像元尺寸会用于 GSD 校验：设置 `resolution_m: null` 可自动推导，
显式设置与实际网格不一致时会报错。以后接入 Landsat/Sentinel 时，继续先按
现有读取器对齐到目标网格；此接口尚不直接接收多个原生分辨率的张量。

## 显存与接缝控制

默认 `patch_size: 32`、`subpatch_size: 4`。320×320 的含 halo 窗口、
9个动态产品加1个静态模态在 combiner 中产生1001个 token（含 CLS）。
默认 `max_tokens: 2048` 超限即报错，不静默降采样。
局部空间注意力另有 `max_subpatch_tokens: 257` 的预算，防止只增大 patch
导致单个 patch 内的注意力矩阵过大。
时间投影按256个空间子块分批，空间 Transformer 按16个 patch 分批，
训练启用梯度检查点。增大模型、窗口或 batch 前需重新测量实际峰值。

Dense 输出拼接融合 patch 特征与选定模态的空间子块特征，经 PixelShuffle
恢复像元分辨率，再用3×3卷积细化。层级概率在最终分辨率计算：
`P(fine) = P(coarse) * P(fine | coarse)`。
继续使用 halo、训练重叠一致性损失、推理高斯融合以及统一类别映射表。
这些机制用于降低接缝风险；最终地图的接缝与精度仍需训练后评估。

## 使用

默认训练（Tiny，随机初始化）：

```bash
uv run python scripts/train.py data/processed/<数据集时间戳>
```

官方 Base 预训练核心（新增投影器与分类头仍需训练）：

```bash
uv run python scripts/download_anysat.py
uv run python scripts/train.py data/processed/<数据集时间戳> --config configs/model_anysat_base.yaml
```

下载器从官方 `g-astruc/AnySat` 获取权重，将解析后的 revision 记入
`third_party/pretrained/anysat/provenance.json`，可用 `--revision` 指定版本。
初始化要求完整共享核心的每个参数名称和形状均匹配，否则报错，不静默跳过。
共享核心默认以 `optimizer.pretrained_learning_rate`（缺省1e-5）微调，
新增层使用普通学习率。训练产物包含全部参数，后续预测或续训无需原始预训练文件。
本次未下载大型 Base 权重，也未测量 Base 的训练显存或精度。

真实窗口检查：

```bash
uv run python scripts/check_anysat_training.py data/processed/<数据集时间戳> --output experiments/anysat_validation/real_window.json
```

该检查执行实测标签损失、AMP反传、重叠一致性反传、AdamW、EMA和验证前向，
不会启动完整训练或写预测地图。一次检查不代表模型已经获得可用分类精度。

本次使用 `20260909_190125` 的一个含实测点窗口，58个时间位置、15个动态特征、
5个静态特征、320×320输入、32类输出。默认 Tiny 共5,955,817个参数，
RTX 2070 SUPER 8GB / FP16 实测峰值张量显存约0.83 GiB，缓存保留约1.14 GiB。
此数值只覆盖该窗口与配置，不是4090实测，也不是整个数据集的显存上界。
