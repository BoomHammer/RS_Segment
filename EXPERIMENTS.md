# 复用预处理数据的实验

所有命令在项目根目录执行，将 `<数据集时间戳>` 替换为已有目录。
这些命令只训练、按需测试和全图预测，不运行预处理、伪标签生成或重新随机划分。

```powershell
# 1. 只用真实标签：不读取伪标签栅格，伪标签损失权重为零
uv run rs-pipeline --retrain data/processed/<数据集时间戳> --no-pseudo-labels

# 2. 已有测试集并入训练，保留验证集，跳过最终测试，直接全图预测
uv run rs-pipeline --retrain data/processed/<数据集时间戳> --train-on-test

# 3. 原有流程：使用原配置的模型与伪标签设置，测试后全图预测
uv run rs-pipeline --retrain data/processed/<数据集时间戳>

# 4–8. 选择模型，正常测试和全图预测
uv run rs-pipeline --retrain data/processed/<数据集时间戳> --model segformer-utae
uv run rs-pipeline --retrain data/processed/<数据集时间戳> --model utae
uv run rs-pipeline --retrain data/processed/<数据集时间戳> --model segformer
uv run rs-pipeline --retrain data/processed/<数据集时间戳> --model maestro
uv run rs-pipeline --retrain data/processed/<数据集时间戳> --model anysat

# 可自由组合
uv run rs-pipeline --retrain data/processed/<数据集时间戳> --model utae --no-pseudo-labels --train-on-test
```

| 参数 | 实现 |
| --- | --- |
| 不传 `--model` | 保留 `--model-config` 指定架构；当前默认为 `lightweight_dual_branch` |
| `segformer-utae` | 完整 SegFormer MiT-B1 静态分支 + U-TAE 时序分支及多尺度融合 |
| `utae` | 单个 U-TAE；动态序列保留时间维度，静态特征附加到每帧 |
| `segformer` | 单个 MiT-B1；动态特征按有效观测取时间均值，与静态特征及有效性掩码拼接 |
| `maestro` | 项目已有 MAESTRO-S 实现 |
| `anysat` | 项目已有 AnySat 实现，默认配置为 tiny |
| `lightweight` | 显式选择原有轻量双分支模型 |

单模型基线仍使用全部输入来源，SegFormer 不将时间帧展平为通道。
所有模型共用层级分类头、真实标签监督、重叠一致性训练及已有全图融合预测流程。
继续使用流式读取、混合精度和梯度累积；U-TAE 按帧分块并使用梯度检查点。
实际峰值显存随时间序列长度、窗口和模型配置变化，未进行完整数据集的 24GB 峰值显存验证。

模型选择不自动下载权重。默认配置下上述模型随机初始化；混合模型若配置了
`model.pretrained.path`，会使用本地 MiT 权重。单独 SegFormer 当前为随机初始化基线。
AnySat 预训练配置可通过 `--model-config configs/model_anysat_base.yaml` 指定。

`--train-on-test` 同时合并已有训练/测试窗口和标签空间块归属，验证集继续隔离用于早停和模型选择。
原始 `data/processed/.../spatial_split.json` 不变，合并视图仅保存到本次实验目录。
测试指标不再生成，此实验不能作为独立测试集成绩使用。

生效后的模型配置、监督策略和合并视图保存在 `experiments/<时间戳>/`。
续训使用 `uv run rs-pipeline --resume experiments/<时间戳>/last.pt`，自动恢复实验设置；
原实验合并了测试集时，续训结束同样跳过测试。
