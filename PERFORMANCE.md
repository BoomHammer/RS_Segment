# 训练性能诊断

默认关闭，不改变模型、损失、采样、halo 或梯度累积策略。
在 Linux 项目根目录给原有命令加 `RS_PROFILE_STEPS=60` 即可；环境变量也适用于
torchrun 自动启动的各个 rank，以及读取旧实验配置的断点续训。

已有数据，从头训练（不重新运行 SAM）：

```bash
RS_PROFILE_STEPS=60 uv run rs-pipeline --retrain data/processed/20261006_162306
```

已有完整训练断点，继续原实验（替换实际路径）：

```bash
RS_PROFILE_STEPS=60 uv run rs-pipeline --resume experiments/<实验目录>/last.pt
```

不能把 `--retrain` 当作续训；它会创建新实验并从头训练。
也可将 `configs/train.yaml` 中 `training.profile_steps` 设为正整数；环境变量优先。
仅诊断训练，测试和全图预测不受影响。

## 输出

每个 rank 在实验目录分别写入 `performance_rank<RANK>_pid<PID>.jsonl`，每个 batch
落盘一行，同时打印 `[PERF rank=...]`。每个 rank 完成指定批次数后停止主进程计时，
训练继续，不自动退出。不增加分布式 barrier 或 gather。
worker 单独对各自前 N×batch_size 个样本计时，因此预取队列可能仍有少量后续诊断数据；
worker 不写文件，不收集栅格内容，只传递小型计时字典。

请提供两个 rank 的 JSONL 文件。无需等完整 epoch；先收集 30–60 批次。
不要仅凭最初一两个 batch 判断，首次迭代包含 worker 启动、缓存预热等成本。
中途退出不会替你保存新断点；没有 `last.pt` 时不要为了诊断丢弃已有训练进度。

## 字段（单位：秒）

| 字段 | 含义 |
|---|---|
| `data_wait_s` | 主进程等待下一批次；含队列、传输及 pin-memory 等等待，不等于 worker 总耗时 |
| `transfer_s` | CPU 到设备传输与等待完成 |
| `forward_s` | 模型前向，可能含 DDP 前向同步等待 |
| `loss_metrics_s` | 损失、监督检查、精度统计及其 CPU/GPU 同步 |
| `backward_s` | 主损失反向，含 DDP 通信及等待其他 rank，**不是纯 GPU 计算时间** |
| `overlap_s` | 重叠一致性分支的前向、损失和反向；未触发时接近零 |
| `optimizer_s` | 优化器、梯度裁剪、调度器及 EMA；未更新时接近零 |
| `cleanup_s` | 最后统计、释放引用、进度条等 |
| `step_s` / `iteration_s` | 本步处理耗时 / 加上取数据等待；不含诊断日志写入 |

epoch 末尾不足一个累积组的额外 optimizer 更新保持原逻辑，在循环外执行，不包含在上述 batch 计时中。

`worker.samples[].seconds` 为每个样本的 worker 耗时：

- `sample_total_s`：整个样本，包括增强。
- `read_window_s`：窗口输入、标签、mask 等构造。
- `read_asset_s`：所有栅格读取加有效值处理和缩放的累计时间。
- `raster_open_read_close_s`：上项内部的借用/打开栅格、读取和归还/关闭；读取可能含解码和重投影。
- `normalize_s`、`missing_fill_s`、`stack_s`：标准化、缺失时相/特征填充、数组堆叠。
- `transforms_s`：数据增强（未配置增强时没有此字段）。
- `worker.collate_s`：组装批次；`worker.dynamic_shape`：实际动态输入 B,T,C,H,W。

上述 worker 字段有嵌套包含关系，不能全部相加；不同 worker 以及预取/训练之间也存在重叠。
例如 `read_asset_s - raster_open_read_close_s` 近似代表读取后的数组转换、mask 和缩放开销。

## 注意

启用时在各阶段边界调用本卡 `torch.cuda.synchronize`，得到包含等待的分段墙钟时间。
这会改变流水重叠、影响吞吐，只用于定位方向，不作为正式速度基准。
关闭后再用稳定阶段的每步耗时验证优化效果。CPU 和单卡也支持；关闭时不调用 CUDA 同步。
诊断不缓存额外输入或模型张量，不增加显存中的大数组。
