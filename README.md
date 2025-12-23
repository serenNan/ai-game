# AI 五子棋

基于深度强化学习和蒙特卡洛树搜索（MCTS）实现的五子棋 AI，通过自我对弈进行训练。

## 项目特点

- 策略-价值网络 + MCTS 搜索算法
- 支持多种深度学习框架：Theano/Lasagne、PyTorch、TensorFlow、Keras
- 纯 NumPy 推理：无需安装深度学习框架即可运行游戏
- Pygame 图形界面：支持人机对战
- 预训练模型：提供 6x6（4子连珠）和 8x8（5子连珠）两种规格

## 环境要求

### 运行游戏（人机对战）

```
Python >= 3.6
NumPy >= 1.11
Pygame
```

### 训练模型

除上述依赖外，还需要以下框架之一：
- Theano >= 0.7 + Lasagne >= 0.1（默认）
- PyTorch >= 0.2.0
- TensorFlow
- Keras

## 安装

```bash
# 安装依赖
pip install numpy pygame

# 如需训练，安装深度学习框架（以 PyTorch 为例）
pip install torch
```

## 运行指令

### 启动游戏（人机对战）

```bash
python main.py
```

启动后会显示菜单界面，可选择：
- **人机对战（先手）**：玩家执黑先行
- **人机对战（后手）**：AI 执黑先行，玩家执白

游戏界面支持：
- 点击棋盘落子
- 重新开始
- 返回菜单

### 训练模型

```bash
python trainer.py
```

默认使用 Theano/Lasagne 框架。如需使用其他框架，修改 `trainer.py` 中的导入：

```python
# 选择一个框架：
from model_theano import NeuralNetworkEvaluator  # Theano/Lasagne（默认）
from model_torch import NeuralNetworkEvaluator   # PyTorch
from model_tf import NeuralNetworkEvaluator      # TensorFlow
from model_keras import NeuralNetworkEvaluator   # Keras
```

训练过程中会自动保存模型：
- `current_policy.model`：当前训练的模型
- `best_policy.model`：评估中表现最好的模型

### 对比模型性能

```bash
python compare_models.py
```

用于对比两个模型的强弱，通过多局对战统计胜率。

---

## 文件详细说明

### 1. main.py - 游戏主程序

**功能**：Pygame 图形界面，提供人机对战功能。

**核心类**：
- `GomokuInterface`：游戏界面管理类

**主要方法**：
| 方法 | 功能 |
|------|------|
| `__init__()` | 初始化 Pygame 窗口、加载字体、创建棋盘状态、加载 AI 模型 |
| `_initializeAI()` | 加载预训练模型文件，创建 AI 代理 |
| `renderMenuScreen()` | 绘制菜单界面（人机对战先手/后手按钮） |
| `renderBoard()` | 绘制棋盘网格线 |
| `renderStones()` | 绘制棋子，高亮最后一步 |
| `renderInfoPanel()` | 绘制底部状态栏和按钮 |
| `resetGame()` | 重置游戏状态，根据模式决定谁先手 |
| `_handleHumanMove(x, y)` | 处理玩家点击落子 |
| `_executeAiMove()` | 在后台线程执行 AI 落子 |
| `_checkGameEnd()` | 检查游戏是否结束（胜负或平局） |
| `run()` | 主循环，处理事件和渲染 |

**工作流程**：
```
启动 → 显示菜单 → 选择模式 → 进入游戏
     ↓
游戏循环：玩家落子 → AI思考 → AI落子 → 检查胜负 → 继续/结束
```

---

### 2. board.py - 游戏引擎

**功能**：棋盘状态管理、规则判定、游戏流程控制。

**核心类**：

#### GameState - 棋盘状态类
| 属性 | 说明 |
|------|------|
| `cols`, `rows` | 棋盘宽高 |
| `positions` | 字典，记录每个位置的棋子（key=位置索引, value=玩家ID） |
| `openPositions` | 列表，可落子的空位 |
| `activePlayer` | 当前执棋方（1或2） |
| `winCondition` | 获胜所需连珠数 |
| `previousMove` | 上一步落子位置 |

| 方法 | 功能 |
|------|------|
| `initState(firstPlayer)` | 初始化棋盘，指定先手玩家 |
| `indexToCoord(idx)` | 将一维索引转为二维坐标 |
| `coordToIndex(coord)` | 将二维坐标转为一维索引 |
| `getStateArray()` | 返回4通道状态数组（用于神经网络输入） |
| `applyMove(moveIdx)` | 执行落子，更新状态 |
| `checkVictory()` | 检查是否有人获胜（横/竖/斜四个方向） |
| `isTerminal()` | 检查游戏是否结束 |

**状态数组格式**（4通道）：
```
通道0: 当前玩家的棋子位置 (1.0表示有棋子)
通道1: 对手的棋子位置
通道2: 上一步落子位置
通道3: 当前执棋方标识 (先手方全1，后手方全0)
```

#### GameController - 游戏控制类
| 方法 | 功能 |
|------|------|
| `renderBoard()` | 在终端打印棋盘（调试用） |
| `runMatch(agent1, agent2)` | 运行一局对战，返回胜者 |
| `runSelfPlay(agent)` | 自我对弈，收集训练数据 |

---

### 3. neural_search.py - MCTS 搜索（神经网络引导）

**功能**：实现带神经网络引导的蒙特卡洛树搜索。

**核心类**：

#### SearchNode - 搜索树节点
| 属性 | 说明 |
|------|------|
| `_parentNode` | 父节点引用 |
| `_childNodes` | 子节点字典（key=动作, value=节点） |
| `_visitCount` | 访问次数 N |
| `_qValue` | 平均价值 Q |
| `_priorProb` | 先验概率 P（来自策略网络） |
| `_ucbBonus` | UCB 探索奖励 U |

| 方法 | 功能 |
|------|------|
| `expandNode(actionPriors)` | 扩展子节点，使用策略网络的概率作为先验 |
| `selectChild(explorationWeight)` | 选择 Q+U 最大的子节点 |
| `updateStats(leafValue)` | 更新节点统计（增量平均） |
| `backpropagate(leafValue)` | 反向传播，更新路径上所有节点 |
| `computeScore(explorationWeight)` | 计算 UCB 分数 |

**UCB 公式**：
```
UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N)

Q: 该节点的平均价值
P: 先验概率（策略网络输出）
N: 该节点访问次数
N_parent: 父节点访问次数
c_puct: 探索系数（默认5）
```

#### MonteCarloTreeSearch - MCTS 搜索类
| 方法 | 功能 |
|------|------|
| `_runSimulation(gameState)` | 执行一次模拟：选择→扩展→评估→回溯 |
| `computeMoveDistribution(gameState, temperature)` | 运行多次模拟，返回动作概率分布 |
| `advanceTree(lastAction)` | 复用子树，将子节点提升为根节点 |

**搜索流程**：
```
1. 选择(Select): 从根节点出发，选择UCB最大的子节点，直到叶节点
2. 扩展(Expand): 使用策略网络扩展叶节点的所有合法动作
3. 评估(Evaluate): 使用价值网络评估叶节点局面
4. 回溯(Backup): 将评估值反向传播，更新路径上所有节点的Q值
```

#### TreeSearchAgent - AI 代理类
| 方法 | 功能 |
|------|------|
| `selectMove(gameState, temperature)` | 选择最佳落子位置 |
| `resetState()` | 重置搜索树 |

**自我对弈模式**：添加 Dirichlet 噪声增加探索：
```python
selectedMove = np.random.choice(
    actions,
    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
)
```

---

### 4. random_search.py - 纯 MCTS（无神经网络）

**功能**：不使用神经网络的传统 MCTS，用于训练时的基准评估。

**与 neural_search.py 的区别**：
| 对比项 | neural_search.py | random_search.py |
|--------|------------------|------------------|
| 扩展策略 | 策略网络输出概率 | 均匀概率 |
| 评估方式 | 价值网络直接评估 | 随机模拟到终局 |
| 速度 | 较快（网络一次前向传播） | 较慢（需要完整模拟） |
| 强度 | 依赖网络质量 | 模拟次数越多越强 |

**随机模拟（Rollout）**：
```python
def _performRollout(self, gameState, maxMoves=1000):
    """随机落子直到游戏结束，返回胜负结果"""
    currentPlayer = gameState.getCurrentPlayer()
    for _ in range(maxMoves):
        if gameState.isTerminal():
            break
        # 随机选择一个合法动作
        actionProbs = randomRolloutPolicy(gameState)
        bestAction = max(actionProbs, key=itemgetter(1))[0]
        gameState.applyMove(bestAction)
    # 返回当前玩家视角的胜负
    return 1 if victor == currentPlayer else -1
```

---

### 5. model_inference.py - NumPy 推理引擎

**功能**：纯 NumPy 实现的神经网络前向传播，用于加载预训练模型进行推理，无需安装深度学习框架。

**核心函数**：
| 函数 | 功能 |
|------|------|
| `convolutionForward()` | 卷积层前向传播 |
| `fullyConnectedForward()` | 全连接层前向传播 |
| `applyRelu()` | ReLU 激活函数 |
| `computeSoftmax()` | Softmax 归一化 |
| `imageToColumns()` | im2col 转换（优化卷积计算） |

**网络结构**：
```
输入: 4 x boardCols x boardRows

共享层:
  Conv1: 4 → 32 通道, 3x3卷积, padding=1
  Conv2: 32 → 64 通道, 3x3卷积, padding=1
  Conv3: 64 → 128 通道, 3x3卷积, padding=1

策略头:
  Conv: 128 → 4 通道, 1x1卷积
  FC: 4*width*height → width*height
  Softmax: 输出每个位置的落子概率

价值头:
  Conv: 128 → 2 通道, 1x1卷积
  FC1: 2*width*height → 64
  FC2: 64 → 1
  Tanh: 输出 [-1, 1] 的局面评估值
```

---

### 6. model_torch.py - PyTorch 实现

**功能**：使用 PyTorch 实现的策略-价值网络，支持 GPU 训练。

**核心类**：

#### ConvolutionalNetwork - 网络结构
```python
class ConvolutionalNetwork(nn.Module):
    def __init__(self, boardCols, boardRows):
        # 共享卷积层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 策略头
        self.policyConv = nn.Conv2d(128, 4, kernel_size=1)
        self.policyFc = nn.Linear(4 * boardCols * boardRows, boardCols * boardRows)
        # 价值头
        self.valueConv = nn.Conv2d(128, 2, kernel_size=1)
        self.valueFc1 = nn.Linear(2 * boardCols * boardRows, 64)
        self.valueFc2 = nn.Linear(64, 1)
```

#### NeuralNetworkEvaluator - 训练封装类
| 方法 | 功能 |
|------|------|
| `batchEvaluate(stateBatch)` | 批量评估状态 |
| `evaluatePosition(gameState)` | 评估单个局面 |
| `trainOnBatch(...)` | 执行一次梯度更新 |
| `saveCheckpoint(filePath)` | 保存模型参数 |

**损失函数**：
```
L = (z - v)² + π^T · log(p) + c||θ||²

z: 实际游戏结果 (+1/-1)
v: 价值网络预测
π: MCTS 搜索得到的概率分布
p: 策略网络输出
c: L2正则化系数
```

---

### 7. model_theano.py - Theano/Lasagne 实现

**功能**：使用 Theano/Lasagne 实现的策略-价值网络（默认训练框架）。

**结构与 model_torch.py 相同**，API 兼容。

---

### 8. trainer.py - 训练入口

**功能**：实现完整的自我对弈训练流程。

**核心类 TrainingManager**：

| 属性 | 默认值 | 说明 |
|------|--------|------|
| `boardCols/boardRows` | 6 | 棋盘尺寸 |
| `winLength` | 4 | 连珠数量 |
| `numSimulations` | 400 | 每步 MCTS 模拟次数 |
| `miniBatchSize` | 512 | 训练批次大小 |
| `replayBufferSize` | 10000 | 经验回放缓冲区大小 |
| `baseLearningRate` | 2e-3 | 基础学习率 |
| `klTarget` | 0.02 | KL 散度目标 |
| `evaluationInterval` | 50 | 评估间隔 |
| `totalBatches` | 1500 | 总训练轮数 |

| 方法 | 功能 |
|------|------|
| `augmentData(gameData)` | 数据增强：旋转+翻转，扩充8倍 |
| `generateSelfPlayData(numGames)` | 自我对弈生成训练数据 |
| `updatePolicy()` | 从缓冲区采样训练网络 |
| `evaluatePolicy(numGames)` | 与纯 MCTS 对战评估强度 |
| `runTraining()` | 主训练循环 |

**训练流程**：
```
初始化网络
↓
循环:
  1. 自我对弈生成数据
  2. 数据增强 (旋转/翻转 8倍)
  3. 存入经验回放缓冲区
  4. 从缓冲区采样训练
  5. 自适应调整学习率 (基于KL散度)
  6. 定期评估 (与纯MCTS对战)
  7. 保存最佳模型
```

**数据增强**：
```python
def augmentData(self, gameData):
    """对每个样本进行4次旋转+翻转，生成8个等价样本"""
    for state, moveProbs, outcome in gameData:
        for rotation in [1, 2, 3, 4]:
            # 旋转状态和概率分布
            rotatedState = np.rot90(state, rotation)
            rotatedProbs = np.rot90(moveProbs, rotation)
            # 水平翻转
            flippedState = np.fliplr(rotatedState)
            flippedProbs = np.fliplr(rotatedProbs)
```

**自适应学习率**：
```python
# 根据KL散度调整学习率
if klDivergence > self.klTarget * 2:
    self.learningRateScale /= 1.5  # KL过大，降低学习率
elif klDivergence < self.klTarget / 2:
    self.learningRateScale *= 1.5  # KL过小，提高学习率
```

---

---

## 预训练模型

| 文件 | 棋盘 | 连珠 | 说明 |
|------|------|------|------|
| `6_6_4.model` | 6x6 | 4子 | 小棋盘，快速体验 |
| `8_8_5.model` | 8x8 | 5子 | 中等棋盘 |

模型格式为 Theano/Lasagne 的 pickle 文件，`model_inference.py` 可直接加载推理。

---

## 训练建议

1. **从小棋盘开始**：建议先用 6x6 + 4子连珠进行训练测试

2. **使用 GPU**：修改 `model_torch.py` 中 `useGpu=True` 加速训练

## 许可证

MIT License
