import numpy as np
import copy


def computeSoftmax(x):
    """计算 softmax 概率分布"""
    probabilities = np.exp(x - np.max(x))
    probabilities /= np.sum(probabilities)
    return probabilities


class SearchNode(object):
    """
    MCTS 搜索树中的节点
    跟踪 Q 值、先验概率 P 和访问调整分数 U
    """

    def __init__(self, parentNode, priorProb):
        self._parentNode = parentNode
        self._childNodes = {}
        self._visitCount = 0
        self._qValue = 0
        self._ucbBonus = 0
        self._priorProb = priorProb

    def expandNode(self, actionPriors):
        """
        通过添加子节点扩展树
        actionPriors: (动作, 先验概率) 元组列表
        """
        for action, prob in actionPriors:
            if action not in self._childNodes:
                self._childNodes[action] = SearchNode(self, prob)

    def selectChild(self, explorationWeight):
        """
        选择 Q + U 值最高的子节点
        返回: (动作, 子节点) 元组
        """
        return max(self._childNodes.items(),
                   key=lambda item: item[1].computeScore(explorationWeight))

    def updateStats(self, leafValue):
        """
        叶节点评估后更新节点统计信息
        leafValue: 从当前玩家视角的评估值
        """
        self._visitCount += 1
        self._qValue += 1.0 * (leafValue - self._qValue) / self._visitCount

    def backpropagate(self, leafValue):
        """递归更新所有祖先节点"""
        if self._parentNode:
            self._parentNode.backpropagate(-leafValue)
        self.updateStats(leafValue)

    def computeScore(self, explorationWeight):
        """
        计算节点分数: Q + U
        explorationWeight: 控制探索与利用的平衡
        """
        self._ucbBonus = (explorationWeight * self._priorProb *
                         np.sqrt(self._parentNode._visitCount) / (1 + self._visitCount))
        return self._qValue + self._ucbBonus

    def isLeafNode(self):
        """检查节点是否没有子节点"""
        return self._childNodes == {}

    def isRootNode(self):
        """检查节点是否为根节点"""
        return self._parentNode is None


class MonteCarloTreeSearch(object):
    """AlphaZero 风格的蒙特卡洛树搜索实现"""

    def __init__(self, policyValueFn, explorationWeight=5, numSimulations=10000):
        """
        policyValueFn: 返回 (动作概率, 价值) 的状态评估函数
        explorationWeight: UCB 探索常数 (c_puct)
        numSimulations: 每步棋的 MCTS 迭代次数
        """
        self._rootNode = SearchNode(None, 1.0)
        self._evaluator = policyValueFn
        self._explorationWeight = explorationWeight
        self._numSimulations = numSimulations

    def _runSimulation(self, gameState):
        """
        执行一次从根节点到叶节点的模拟，评估叶节点，反向传播
        gameState 会被原地修改；调用者必须提供副本
        """
        node = self._rootNode
        while True:
            if node.isLeafNode():
                break
            action, node = node.selectChild(self._explorationWeight)
            gameState.applyMove(action)

        actionProbs, leafValue = self._evaluator(gameState)
        finished, victor = gameState.isTerminal()
        if not finished:
            node.expandNode(actionProbs)
        else:
            if victor == -1:
                leafValue = 0.0
            else:
                leafValue = (
                    1.0 if victor == gameState.getCurrentPlayer() else -1.0
                )

        node.backpropagate(-leafValue)

    def computeMoveDistribution(self, gameState, temperature=1e-3):
        """
        运行所有模拟并返回动作概率分布
        temperature: 控制动作选择的探索程度
        """
        for _ in range(self._numSimulations):
            stateCopy = copy.deepcopy(gameState)
            self._runSimulation(stateCopy)

        actionVisits = [(action, node._visitCount)
                        for action, node in self._rootNode._childNodes.items()]
        actions, visits = zip(*actionVisits)
        actionProbs = computeSoftmax(1.0 / temperature * np.log(np.array(visits) + 1e-10))

        return actions, actionProbs

    def advanceTree(self, lastAction):
        """
        将根节点移动到对应 lastAction 的子节点
        保留子树的搜索知识
        """
        if lastAction in self._rootNode._childNodes:
            self._rootNode = self._rootNode._childNodes[lastAction]
            self._rootNode._parentNode = None
        else:
            self._rootNode = SearchNode(None, 1.0)

    def __str__(self):
        return "MonteCarloTreeSearch"


class TreeSearchAgent(object):
    """使用神经网络引导 MCTS 的 AI 玩家"""

    def __init__(self, policyValueFn,
                 explorationWeight=5, numSimulations=2000, selfPlayMode=0):
        self.searchTree = MonteCarloTreeSearch(policyValueFn, explorationWeight, numSimulations)
        self._selfPlayMode = selfPlayMode

    def assignPlayerId(self, playerId):
        """分配玩家 ID"""
        self.playerId = playerId

    def resetState(self):
        """重置搜索树状态"""
        self.searchTree.advanceTree(-1)

    def selectMove(self, gameState, temperature=1e-3, returnProb=0):
        """选择最佳落子位置"""
        validMoves = gameState.openPositions
        moveProbabilities = np.zeros(gameState.cols * gameState.rows)
        if len(validMoves) > 0:
            actions, probs = self.searchTree.computeMoveDistribution(gameState, temperature)
            moveProbabilities[list(actions)] = probs
            if self._selfPlayMode:
                # 自我对弈时添加 Dirichlet 噪声以增加探索
                selectedMove = np.random.choice(
                    actions,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                self.searchTree.advanceTree(selectedMove)
            else:
                selectedMove = np.random.choice(actions, p=probs)
                self.searchTree.advanceTree(-1)

            if returnProb:
                return selectedMove, moveProbabilities
            else:
                return selectedMove
        else:
            print("警告: 没有可用的合法落子位置")

    def __str__(self):
        return "TreeSearchAgent {}".format(self.playerId)
