# 训练指南

本文档详细介绍如何从零开始训练五子棋 AI 模型。

## 环境准备

### 安装依赖

```bash
# 基础依赖
pip install numpy

# 选择一个深度学习框架（推荐 PyTorch）
pip install torch

# 或使用其他框架
pip install tensorflow  # TensorFlow
pip install keras       # Keras
```

### 修改框架配置

编辑 `trainer.py`，修改导入语句：

```python
# 默认使用 Theano（注释掉）
# from model_theano import NeuralNetworkEvaluator

# 选择你安装的框架（取消注释）
from model_torch import NeuralNetworkEvaluator     # PyTorch
# from model_tf import NeuralNetworkEvaluator      # TensorFlow
# from model_keras import NeuralNetworkEvaluator   # Keras
```

---

## 训练参数详解

所有参数在 `trainer.py` 的 `TrainingManager.__init__` 方法中配置。

### 棋盘配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `boardCols` | 6 | 棋盘列数（宽度） |
| `boardRows` | 6 | 棋盘行数（高度） |
| `winLength` | 4 | 获胜所需连珠数 |

**建议**：
- 初次训练使用 6x6 + 4子，训练速度快，便于验证
- 8x8 + 5子 需要更多训练时间
- 标准五子棋 15x15 需要大量计算资源

```python
self.boardCols = 6
self.boardRows = 6
self.winLength = 4
```

---

### MCTS 搜索参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `numSimulations` | 400 | 每步棋的 MCTS 模拟次数 |
| `explorationWeight` | 5 | UCB 探索系数 (c_puct) |
| `explorationTemp` | 1.0 | 动作选择温度参数 |

**numSimulations（模拟次数）**：
- 值越大，每步搜索越深入，棋力越强
- 值越大，每步耗时越长
- 建议范围：200-800

```python
self.numSimulations = 400
```

**explorationWeight（探索系数）**：
- 控制探索与利用的平衡
- 值越大，越倾向于尝试新走法
- 值越小，越倾向于选择已知好的走法
- 建议范围：3-8

```python
self.explorationWeight = 5
```

**explorationTemp（温度参数）**：
- 控制自我对弈时动作选择的随机性
- 值越大，选择越随机（增加探索）
- 值越小，越倾向于选择最优动作
- 建议：1.0（标准）

```python
self.explorationTemp = 1.0
```

---

### 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `baseLearningRate` | 2e-3 | 基础学习率 |
| `learningRateScale` | 1.0 | 学习率缩放因子（自适应调整） |
| `l2Regularization` | 1e-4 | L2 正则化系数 |
| `trainingEpochs` | 5 | 每次更新的训练轮数 |
| `klTarget` | 0.02 | KL 散度目标值 |

**baseLearningRate（基础学习率）**：
- 控制每次梯度更新的步长
- 值过大：训练不稳定
- 值过小：收敛缓慢
- 建议范围：1e-4 ~ 5e-3

```python
self.baseLearningRate = 2e-3
```

**klTarget（KL 散度目标）**：
- 用于自适应调整学习率
- 实际 KL > 2 * klTarget：学习率减半
- 实际 KL < 0.5 * klTarget：学习率加倍
- 建议：0.02

```python
self.klTarget = 0.02
```

---

### 经验回放参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `replayBufferSize` | 10000 | 经验回放缓冲区大小 |
| `miniBatchSize` | 512 | 每次训练的批次大小 |
| `gamesPerBatch` | 1 | 每轮自我对弈的局数 |

**replayBufferSize（缓冲区大小）**：
- 存储最近的训练样本
- 值越大，样本多样性越好，但内存占用越大
- 每局游戏约产生 棋盘大小 * 8 个样本（数据增强后）
- 建议：5000-20000

```python
self.replayBufferSize = 10000
```

**miniBatchSize（批次大小）**：
- 每次梯度更新使用的样本数
- 值越大，梯度估计越稳定，但显存占用越大
- 必须小于 replayBufferSize
- 建议：256-1024

```python
self.miniBatchSize = 512
```

---

### 训练控制参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `totalBatches` | 1500 | 总训练轮数 |
| `evaluationInterval` | 50 | 评估间隔（每 N 轮评估一次） |
| `baselineSimulations` | 1000 | 评估对手的 MCTS 模拟次数 |

**totalBatches（总轮数）**：
- 训练的总迭代次数
- 每轮包含：自我对弈 → 训练 → (可能的评估)
- 建议：1000-3000

```python
self.totalBatches = 1500
```

**evaluationInterval（评估间隔）**：
- 每隔多少轮与纯 MCTS 对战评估
- 评估较慢，间隔太小会拖慢训练
- 建议：50-100

```python
self.evaluationInterval = 50
```

---

## 开始训练

```bash
python trainer.py
```

### 训练输出说明

```
batch:1, episode_length:25
kl:0.00234, lr_scale:1.000, loss:4.523, entropy:3.891, explained_var_prev:0.012, explained_var_new:0.089
```

| 指标 | 说明 |
|------|------|
| `batch` | 当前训练轮数 |
| `episode_length` | 上一局自我对弈的步数 |
| `kl` | KL 散度，衡量策略更新幅度 |
| `lr_scale` | 当前学习率缩放因子 |
| `loss` | 总损失值（越小越好） |
| `entropy` | 策略熵（越大表示探索越多） |
| `explained_var_prev` | 更新前的价值解释方差 |
| `explained_var_new` | 更新后的价值解释方差（应逐渐接近1） |

### 评估输出

```
current batch: 50
baseline_simulations:1000, wins: 8, losses: 1, draws: 1
New best policy found!
```

表示与 1000 次模拟的纯 MCTS 对战 10 局，胜 8 负 1 平 1。

---

## 模型保存

训练过程中自动保存两个模型：

| 文件 | 说明 |
|------|------|
| `current_policy.model` | 每次评估时保存的当前模型 |
| `best_policy.model` | 评估中胜率最高的模型 |

---

## 使用训练好的模型

1. 将 `best_policy.model` 重命名为 `{宽}_{高}_{连珠数}.model`

```bash
# 例如 6x6 棋盘，4子连珠
mv best_policy.model 6_6_4.model
```

2. 修改 `main.py` 中的棋盘配置：

```python
gui = GomokuInterface(boardCols=6, boardRows=6, winLength=4)
```

3. 运行游戏：

```bash
python main.py
```

---

## 训练技巧

### 1. 从小棋盘开始

先用 6x6 + 4子验证训练流程，确认可以正常运行后再尝试更大棋盘。

### 2. 监控 loss 和 entropy

- `loss` 应该逐渐下降
- `entropy` 初期较高（探索多），后期逐渐降低（策略收敛）
- 如果 `entropy` 降到很低但胜率不高，可能陷入局部最优

### 3. 观察 explained_var

- `explained_var_new` 应该逐渐增加
- 接近 1.0 表示价值网络预测准确
- 如果长期接近 0，说明价值网络没有学到有效信息

### 4. 调整学习率

- 如果 `kl` 持续过大（> 0.1），降低 `baseLearningRate`
- 如果 `kl` 持续过小（< 0.001），提高 `baseLearningRate`

### 5. 使用 GPU 加速

修改 `model_torch.py` 中的参数：

```python
self.network = ConvolutionalNetwork(boardCols, boardRows).cuda()  # 使用 GPU
self.useGpu = True
```

---

## 常见问题

### Q: 训练很慢怎么办？

1. 减少 `numSimulations`（如 200）
2. 使用 GPU
3. 使用更小的棋盘

### Q: 模型不收敛？

1. 检查学习率是否合适
2. 增加 `replayBufferSize`
3. 增加 `numSimulations`

### Q: 如何继续训练已有模型？

修改 `trainer.py` 最后的代码：

```python
if __name__ == '__main__':
    # 从已有模型继续训练
    manager = TrainingManager(modelPath='best_policy.model')
    manager.runTraining()
```

### Q: 训练中断了怎么办？

模型每 `evaluationInterval` 轮保存一次，可以从 `current_policy.model` 继续训练。
