# 项目开发状态

## 当前阶段

**阶段 5：模型推理开发。**

训练、验证、测试和基于 checkpoint 的全图无缝预测闭环已经具备。

## 阶段状态总览

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 阶段 1 | 数据准备与弱监督标签生成 | 已完成 |
| 阶段 2 | 数据加载与多分辨率融合 | 已完成 |
| 阶段 3 | SegFormer-U-TAE 模型、训练与验证 | 已完成 |
| 阶段 4 | 层级化损失与长尾类别处理 | 已完成 |
| 阶段 5 | 重叠滑窗无缝推理与全图预测 | 已完成 |

## 阶段 1：数据准备与弱监督标签生成

阶段 1 已完成：

- [x] YAML 配置、数据目录检查、随机种子和基础工具。
- [x] GeoTIFF 分块流式统计、NoData/NaN/Inf 排除和 JSON 缓存。
- [x] 标签 CSV 流式读取、字段映射、坐标/类别校验和层级编码。
- [x] PointSAM、NPC prompt、SAM2.1 small 后端和弱标签 GeoTIFF 生成。
- [x] 弱标签质量评估、类别分布、样点状态和可视化报告。

统一入口：

```bash
uv run python scripts/preprocess.py
```

可选优化：编译 SAM2 CUDA 扩展，以及继续评估光谱回退标签质量；二者不阻塞后续模型开发。

## 阶段 2：数据加载与多分辨率融合

阶段 2 已完成并通过合成数据、真实窗口和全量测试验证：

- [x] 统一动态影像、静态影像、实测样点和弱标签的数据索引协议。
- [x] TorchGeo 窗口级懒加载，多分辨率对齐和流式栅格读取。
- [x] 多时相分组、特征顺序、缺帧处理、时间编码和批内 padding。
- [x] 有效性掩码、NoData/NaN/Inf 处理、归一化和同步空间增强。
- [x] 空间块级 train/validation/test 划分，避免相邻窗口泄漏。
- [x] 基于地面实测类别的分层空间块划分，降低验证集和测试集类别缺失风险。
- [x] DataLoader、采样器、长尾类别权重和 4090 显存/吞吐配置。

入口：

```bash
uv run python scripts/datasets.py data/processed/<YYYYMMDD_HHMMSS>
```

## 阶段 3/4：模型训练与测试

训练和测试功能已实现，可以进行多次独立训练比较：

- [x] SegFormer-U-TAE 风格多尺度模型、动态时序编码和静态特征融合。
- [x] 层级 coarse/fine 输出、实测标签与弱标签掩码监督。
- [x] 伪标签仅参与训练；验证和测试只使用地面实测标签。
- [x] AdamW、Linear Warmup + Cosine Decay、BF16 AMP、梯度累积。
- [x] Gradient Clipping、EMA、Dropout、DropPath 和 validation early stopping。
- [x] 每轮保存 loss、accuracy、validation loss/accuracy、学习率等训练历史。
- [x] 测试输出 fine/coarse 的 Accuracy、Precision、Recall、F1、IoU、ROC/AUC、MSE、MAE 和混淆矩阵。

训练命令：

```bash
uv run python scripts/train.py data/processed/<YYYYMMDD_HHMMSS>
```

每次训练按照训练启动时间创建独立目录：

```text
experiments/<训练开始时间>/
├── model_<训练开始时间>.pt
├── train_log.json
├── train.yaml
├── model.yaml
├── data.yaml
└── test_metrics.json       # 执行测试后生成
```

测试命令：

```bash
uv run python scripts/test.py experiments/<训练开始时间>/model_<训练开始时间>.pt
```

`train_log.json` 保存训练数据来源、配置快照、逐 epoch 历史、最佳模型和早停信息，保证 checkpoint 与数据集对应。

## 阶段 5：模型推理开发

全图预测功能已实现，入口为 [scripts/predict.py](scripts/predict.py)，配置文件为 [configs/predict.yaml](configs/predict.yaml)。

- 已实现内容：

1. 自动读取 checkpoint 对应的 `train_log.json`、`data.yaml`、统计量和类别映射。
2. 重叠滑窗、边界窗口处理和内存映射流式推理。
3. 使用高斯权重融合窗口结果，降低接缝和方块效应。
4. 输出带 CRS、transform、nodata 和类别映射信息的 GeoTIFF。
5. 通过 `override_ground_truth` 配置选择是否用地面实测标签覆盖输出像素。

预测命令：

```bash
uv run python scripts/predict.py \
  experiments/<训练开始时间>/model_<训练开始时间>.pt \
  --config configs/predict.yaml
```

## 验证状态

- [x] `pytest`：44 项通过；仅有 rasterio 的弃用提示。
- [x] `ruff check .`：通过。
- [x] 训练入口已完成真实数据单 epoch 冒烟训练。
- [x] 测试入口已完成 checkpoint 加载和测试指标输出验证。
- [x] 预测入口及全图无缝推理已实现，并支持地面标签覆盖开关。
- [ ] 多 epoch 训练效果和最终制图质量尚待实验验收。
