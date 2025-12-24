"""
纯蒙特卡洛树搜索实现（不使用神经网络）
使用随机模拟进行局面评估
"""

import numpy as np
import copy
from operator import itemgetter


def randomRolloutPolicy(gameState):
    """随机策略用于模拟阶段"""
    actionProbs = np.random.rand(len(gameState.openPositions))
    return zip(gameState.openPositions, actionProbs)


def uniformEvaluator(gameState):
    """返回均匀动作概率和零价值"""
    numActions = len(gameState.openPositions)
    actionProbs = np.ones(numActions) / numActions
    return zip(gameState.openPositions, actionProbs), 0


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


class PureMonteCarloSearch(object):
    """不使用神经网络引导的纯 MCTS"""

    def __init__(self, policyValueFn, explorationWeight=5, numSimulations=10000):
        """
        policyValueFn: 返回 (动作概率, 价值) 的状态评估函数
        explorationWeight: UCB 探索常数
        numSimulations: 每步棋的 MCTS 迭代次数
        """
        self._rootNode = SearchNode(None, 1.0)
        self._evaluator = policyValueFn
        self._explorationWeight = explorationWeight
        self._numSimulations = numSimulations

    def _runSimulation(self, gameState):
        """
        执行一次从根节点到叶节点的模拟，通过随机模拟评估，反向传播
        gameState 会被原地修改；调用者必须提供副本
        """
        node = self._rootNode
        while True:
            if node.isLeafNode():
                break
            action, node = node.selectChild(self._explorationWeight)
            gameState.applyMove(action)

        actionProbs, _ = self._evaluator(gameState)
        finished, victor = gameState.isTerminal()
        if not finished:
            node.expandNode(actionProbs)
        leafValue = self._performRollout(gameState)
        node.backpropagate(-leafValue)

    def _performRollout(self, gameState, maxMoves=1000):
        """
        随机落子直到游戏结束
        返回: 当前玩家赢返回 +1，输返回 -1，平局返回 0
        """
        currentPlayer = gameState.getCurrentPlayer()
        for _ in range(maxMoves):
            finished, victor = gameState.isTerminal()
            if finished:
                break
            actionProbs = randomRolloutPolicy(gameState)
            bestAction = max(actionProbs, key=itemgetter(1))[0]
            gameState.applyMove(bestAction)
        else:
            print("警告: 随机模拟达到步数上限")
        if victor == -1:
            return 0
        else:
            return 1 if victor == currentPlayer else -1

    def selectBestMove(self, gameState):
        """运行模拟并返回访问次数最多的动作"""
        for _ in range(self._numSimulations):
            stateCopy = copy.deepcopy(gameState)
            self._runSimulation(stateCopy)
        return max(self._rootNode._childNodes.items(),
                   key=lambda item: item[1]._visitCount)[0]

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
        return "PureMonteCarloSearch"


class PureSearchAgent(object):
    """不使用神经网络的纯 MCTS AI 玩家"""

    def __init__(self, explorationWeight=5, numSimulations=2000):
        self.searchTree = PureMonteCarloSearch(uniformEvaluator, explorationWeight, numSimulations)

    def assignPlayerId(self, playerId):
        """分配玩家 ID"""
        self.playerId = playerId

    def resetState(self):
        """重置搜索树状态"""
        self.searchTree.advanceTree(-1)

    def selectMove(self, gameState):
        """选择最佳落子位置"""
        validMoves = gameState.openPositions
        if len(validMoves) > 0:
            move = self.searchTree.selectBestMove(gameState)
            self.searchTree.advanceTree(-1)
            return move
        else:
            print("警告: 没有可用的合法落子位置")

    def __str__(self):
        return "PureSearchAgent {}".format(self.playerId)
