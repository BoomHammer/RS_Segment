# 模型模块说明 Model Module Instructions
### 职责Responsibility
src/losses/
负责损失函数相关
它不得包含模型网络结构、前向传播、参数更新或数据预处理的逻辑

#### 层级化损失函数

#### 类别不平衡 (Class Imbalance)
Focal Loss 仅可作为辅助选项使用，不要完全依赖 Focal Loss。

