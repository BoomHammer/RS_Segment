### 1. 项目介绍 (Project Introduction)
本项目是一个基于弱监督学习的大陆级遥感植被语义分割任务。核心目标是利用极度稀疏的实测样点数据，结合多源异构的遥感影像（多时态动态影像与静态影像），通过深度学习混合架构（Swin-U-TAE）与弱监督标签生成（PointSAM），最终输出覆盖整个研究区域的无缝、无方块效应的250m分辨率植被类型分布图（共8大类，72小类）。

### 2. 环境与工具 (Environment & Tooling)
CUDA 版本: 12.4  
包管理器: uv  
代码规范与检查: Ruff (强制用于所有的代码格式化与 Lint 检查)

### 3. 项目结构 (Project Structure)
├── .venv/  
├── .gitignore  
├── AGENTS.md  
├── configs/   # 存放.yaml文件，管理模型超参数  
├── data/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── labels/  # 标签数据  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── raw/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── dynamic/  # 动态遥感数据  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── static/  # 静态遥感数据  
├── scripts/  # 程序入口（train.py, test.py）  
├── src/  # 源码  
└── test/  # 存储每次实验的模型、结果  

### 4. 项目背景与约束 (Context & Constraints)
**硬件环境:** 本地单机部署，单张 RTX 4090 显卡 (24GB 显存)。

**数据规模:** 初始 100GB，未来扩展至 5TB，必须采用流式/动态读取。

**标签数据:** 极度稀疏的样点数据（约2000个，未来扩展至20000个），CSV格式（经度、纬度、地表类型），呈极度长尾分布。地表类型共72个细分类别，隶属于8个大类。

**输入特征 (多源异构):**

**动态数据:** MODIS多光谱（未来将增加Landsat、Sentinel）、NDVI、温度等时间序列（覆盖2023全年，12-46帧不定长，空间分辨率250m-1km不等）。

**静态数据:** DEM 地形数据（90m分辨率，无时间维度）。温度降水年均数据。

### 5. 核心技术栈 (Core Tech Stack)
**基础框架:** PyTorch, TorchGeo

**模型架构:** PointSAM (Segment Anything Model), Swin Transformer (Tiny), L-TAE (Lightweight Temporal Attention Encoder)

**显存优化:** BF16 混合精度训练 (AMP), 梯度累积 (Gradient Accumulation)

### 6. 严格行为准则 (Strict Directives for Agent)
1. 任何提供的代码必须考虑到 RTX 4090 24GB 的显存瓶颈。
2. 涉及全图推断的代码，必须使用重叠滑窗和高斯加权，拒绝提供简单的分块拼接代码。始终使用全局统计常量。以保证最终输出结果（tiff影像）中没有明显接缝和方格。
3. 所有提供的 Python 代码必须遵循 Ruff 的规范。
