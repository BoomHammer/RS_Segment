# MAESTRO-S 数据适配

默认模型由 `configs/model.yaml` 的 `architecture: maestro_s` 选择。
训练、测试、预测仍使用原来的命令和 checkpoint contract。

## 结构与来源

实现依据 [论文 v2 §3、§4.4](https://arxiv.org/html/2508.10894v2) 和
[官方 Small 配置](https://github.com/IGNF/MAESTRO/blob/main/maestro/ssl/mae.py)。
每组 9 层独立时空 Transformer，随后 3 层跨组 Transformer；宽度 384、
6 个注意力头、MLP 比例 2。各空间位置通过可学习查询池化模态/日期 token，
线性预测 patch 内每个像素，恢复到原窗口尺寸，再计算层级概率。

默认分组是本项目的数据适配选择：

| 组 | 产品 | token 化 |
| --- | --- | --- |
| optical | SR、NDVI、EVI、FPAR、GPP、LAI | 产品独立 tokenizer；SR 的波段联合投影 |
| environment | LST、PR、SOIL 及其他未显式归组的动态产品 | 产品独立 tokenizer |
| static | 索引中所有静态层 | 单时相联合投影 |

新传感器应通过 `maestro.modalities` 显式指定分组，不会被自动识别为光学产品。
例如每项可写成 `{name: sr, role: dynamic, group: optical,
features: [SR_B1, SR_B2], temporal_bins: 4}`。
完整列表须恰好覆盖所有选定输入特征一次。通道数、类别数和父子类关系来自数据产物，
不要求固定为某个数量；`data.stage2.features` 的子集和动态顺序同样参与模型契约。

## 已有数据如何进入模型

- 继续按窗口流式读取，并复用已有 DN 有效范围、Scale、统计标准化和网格对齐。
  不额外应用论文中其他传感器的归一化常数。
- 每个动态产品仅在自己的有效观测中分箱，避免将月尺度 PR 与 8/16 天产品
  强行当作同时采集。训练随机选帧，评估选最接近像元中位数的真实观测；
  短序列空位有明确掩码，不复制日期补齐。
- 每像元缺测、缺波段、缺帧、NaN/Inf 与批内 padding 均被屏蔽；额外的有效性
  通道区分缺测和真实物理零。整组缺失也不会使注意力产生 NaN。
- 时间编码采用年内正弦/余弦、零时刻的小时编码、相对 2023-01-01 的年间偏移。
  月度记录沿用数据层的每月第一天约定；静态输入不伪造采集日期。
- 当前接口已经把所有栅格对齐到共同目标网格，因此空间编码使用相同 token 网格，
  不把重采样后的 DEM 错当成仍有原生 90m 分辨率。
- 特征按名称绑定，静态名称随 batch 传递，防止索引顺序或动态通道重排错配。
- 输出继续使用 `coarse_logits`、`fine_logits`、概率和 `valid_mask`。
  小类概率满足 `P(fine)=P(coarse)P(fine|coarse)`，实测与弱标签仍分别加权。

## 显存与接缝

默认 256×256 核心加每侧 32 像素 halo，最大输入 320×320。
patch_size=32 时，当前 9 种动态产品各选 4 帧、静态 1 帧，共
`(9×4+1)×10×10=3700` tokens。使用 SDPA、梯度检查点、AMP 和梯度累积；
超过 4096 token 即报错，需调整窗口、patch_size 或时间分箱数。
该预算不是任意 batch_size 下的显存保证。

训练启用已有的重叠 KL 一致性损失；预测沿用 halo 中心裁剪与重叠高斯融合，
类别映射和图例仍来自同一标签表。patch 网格并不改变输出栅格分辨率，
但大 patch/少时间分箱可能影响细节和物候精度，需通过空间测试集及接缝检查评估。

真实窗口检查（一次优化步骤，不启动完整训练）：

```bash
uv run python scripts/check_maestro_training.py data/processed/<数据集时间戳> --output experiments/maestro_preflight/report.json
```

此脚本检查实测标签损失、AMP 反向、重叠损失、AdamW、EMA、评估前向和显存峰值。

## 与原论文训练流程的边界

这是 MAESTRO-S **下游分割架构的本地适配实现**，默认随机初始化。
没有添加论文的自监督预训练阶段、75% 掩码重建或 patch-group-wise 重建目标归一化；
这些属于预训练目标，而不是下游分割的必需前向步骤。不能把该实现的随机初始化
训练等同于论文中完成自监督预训练的结果。

本实现增加了缺测有效性输入及项目原有层级分类头，使用项目自己的状态字典命名。
不直接兼容官方权重；官方 README 列出的预训练模型为 Base，不能加载到 Small。
MiT/U-TAE 权重同样不能用于 MAESTRO-S。旧实验通过保留的旧模型类读取和续训，
新实验须重新训练。任何效果提升或无接缝结果均须在训练后实测，不能由结构保证。
