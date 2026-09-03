# 模型模块说明 Model Module Instructions
### 职责Responsibility
src/models/ 负责： 
网络架构定义与构建
子模块组装
激活与归一化
模型变体配置
与架构相关的计算
它不得包含数据加载、预处理、损失函数的逻辑

### 模型架构
使用双分支时空架构 (dual-branch spatio-temporal architecture)
#### 静态分支 (Static branch)
使用模态主干 (Modality Stem) -> SegFormer MiT-B1 编码器。处理DEM、坡度、坡向、年均值等静态或缓慢变化的变量。
#### 动态分支 (Dynamic branch)
使用带有 L-TAE 时间注意力的 U-TAE 风格编码器。处理反射率、NDVI/EVI、LAI/GPP、降水等动态变量。
不要将所有时间戳展平 (flatten) 为普通通道。
#### 多尺度融合 (Multi-scale fusion)
融合对应的静态和动态特征，使用基于门控 (gated) 或基于注意力 (attention-based) 的特征融合。
将融合后的特征传递给一个共享的分割解码器。

### 模型特点
#### 层级分类 (Hierarchical Classification)
使用联合层级分类。共享特征 -> 粗分类头 + 细分类条件头 (Coarse Head + Fine Conditional Heads)

细分类头 (Fine heads)为每个粗分类创建一个细分类专家 (fine-class expert)。

最终细分类概率如下：
P(fine) = P(coarse) * P(fine | coarse)
训练必须直接使用 ground-truth 层级结构。
在训练期间，不要简单地使用粗分类的 argmax 来选择细分类头。
在大类和小类预测之间添加层级一致性监督 (hierarchy-consistency supervision)。

#### 类别不平衡 (Class Imbalance)
使用类别感知采样 (class-aware sampling)。罕见类别必须更频繁地出现，但不能过度重复单个样本。
联合训练粗分类和细分类。

#### 地面调查样点+SAM生成的伪标签(Field truth labels + SAM weak-labels)
地面调查样点是高置信度的监督，SAM 伪标签是含噪监督。绝不要将这两种来源视为同等可靠。
两者需要使用不同的权重。

#### 无缝训练与无缝推理 (Seamless Training and Seamless Inference)
必须在训练期间强制实现无缝性，不要试图仅在推理期间修复 patch 伪影。
最终输出的类别栅格不能包含人为的：patch 边界、笔直的瓦片接缝线、接缝 (seams)、块状伪影 (block artifacts)。


### 模型结构
                           ┌────────────────────────┐
                           │    STATIC FEATURES     │
                           │                        │
                           │ DEM / slope / soil ... │
                           └────────────┬───────────┘
                                        │
                            Modality Stem
                                        │
                                        ▼
                            SegFormer MiT-B1
                                        │
                                S1 S2 S3 S4
                                        │
                                        │
                                        │
        dynamic time series             │
                │                       │
                ▼                       │
        spatial encoder                 │
                │                       │
            L-TAE                       │
                │                       │
        D1 D2 D3 D4                     │
                │                       │
                └──────────┬────────────┘
                           ▼
                Multi-scale gated fusion

                S1 ↔ D1
                S2 ↔ D2
                S3 ↔ D3
                S4 ↔ D4

                        │
                        ▼
                    Shared Decoder
                        │
                        ▼
                    Feature Map F
                        │
                ┌───────┴────────┐
                │                │
                ▼                ▼

            Coarse head       Fine experts

            ~10 classes     ~10 expert heads
                                │
                            total ~70 classes

                │                │
                └───────┬────────┘
                        ▼

                hierarchical fusion

                P(fine)=
                P(coarse)
                ×P(fine|coarse)

                        │
                        ▼
                    ~70-class map