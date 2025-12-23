"""
纯蒙特卡洛树搜索实现（不使用神经网络）
使用随机模拟进行局面评估
"""

import numpy as np
import copy
from operator import itemgetter


def randomRolloutPolicy(gameState):
    """Simple random policy for rollout phase"""
    actionProbs = np.random.rand(len(gameState.openPositions))
    return zip(gameState.openPositions, actionProbs)


def uniformEvaluator(gameState):
    """Returns uniform action probabilities and zero value"""
    numActions = len(gameState.openPositions)
    actionProbs = np.ones(numActions) / numActions
    return zip(gameState.openPositions, actionProbs), 0


class SearchNode(object):
    """
    A node in the MCTS search tree.
    Tracks Q-value, prior probability P, and visit-adjusted score U.
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
        Expand tree by adding child nodes.
        actionPriors: list of (action, prior_probability) tuples
        """
        for action, prob in actionPriors:
            if action not in self._childNodes:
                self._childNodes[action] = SearchNode(self, prob)

    def selectChild(self, explorationWeight):
        """
        Select child with highest Q + U value.
        Returns: (action, child_node) tuple
        """
        return max(self._childNodes.items(),
                   key=lambda item: item[1].computeScore(explorationWeight))

    def updateStats(self, leafValue):
        """
        Update node statistics after leaf evaluation.
        leafValue: evaluation from current player's perspective
        """
        self._visitCount += 1
        self._qValue += 1.0 * (leafValue - self._qValue) / self._visitCount

    def backpropagate(self, leafValue):
        """Recursively update all ancestors"""
        if self._parentNode:
            self._parentNode.backpropagate(-leafValue)
        self.updateStats(leafValue)

    def computeScore(self, explorationWeight):
        """
        Calculate node score: Q + U
        explorationWeight: controls exploration vs exploitation
        """
        self._ucbBonus = (explorationWeight * self._priorProb *
                         np.sqrt(self._parentNode._visitCount) / (1 + self._visitCount))
        return self._qValue + self._ucbBonus

    def isLeafNode(self):
        """Check if node has no children"""
        return self._childNodes == {}

    def isRootNode(self):
        return self._parentNode is None


class PureMonteCarloSearch(object):
    """Pure MCTS without neural network guidance"""

    def __init__(self, policyValueFn, explorationWeight=5, numSimulations=10000):
        """
        policyValueFn: function that returns (action_probs, value) for a state
        explorationWeight: UCB exploration constant
        numSimulations: number of MCTS iterations per move
        """
        self._rootNode = SearchNode(None, 1.0)
        self._evaluator = policyValueFn
        self._explorationWeight = explorationWeight
        self._numSimulations = numSimulations

    def _runSimulation(self, gameState):
        """
        Execute one simulation from root to leaf, evaluate via rollout, backpropagate.
        gameState is modified in-place; caller must provide a copy.
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
        Play random moves until game ends.
        Returns: +1 if current player wins, -1 if loses, 0 if draw
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
            print("WARNING: rollout reached move limit")
        if victor == -1:
            return 0
        else:
            return 1 if victor == currentPlayer else -1

    def selectBestMove(self, gameState):
        """Run simulations and return most visited action"""
        for _ in range(self._numSimulations):
            stateCopy = copy.deepcopy(gameState)
            self._runSimulation(stateCopy)
        return max(self._rootNode._childNodes.items(),
                   key=lambda item: item[1]._visitCount)[0]

    def advanceTree(self, lastAction):
        """
        Move root to child corresponding to lastAction.
        Preserves subtree knowledge.
        """
        if lastAction in self._rootNode._childNodes:
            self._rootNode = self._rootNode._childNodes[lastAction]
            self._rootNode._parentNode = None
        else:
            self._rootNode = SearchNode(None, 1.0)

    def __str__(self):
        return "PureMonteCarloSearch"


class PureSearchAgent(object):
    """AI player using pure MCTS without neural network"""

    def __init__(self, explorationWeight=5, numSimulations=2000):
        self.searchTree = PureMonteCarloSearch(uniformEvaluator, explorationWeight, numSimulations)

    def assignPlayerId(self, playerId):
        self.playerId = playerId

    def resetState(self):
        self.searchTree.advanceTree(-1)

    def selectMove(self, gameState):
        validMoves = gameState.openPositions
        if len(validMoves) > 0:
            move = self.searchTree.selectBestMove(gameState)
            self.searchTree.advanceTree(-1)
            return move
        else:
            print("WARNING: no valid moves available")

    def __str__(self):
        return "PureSearchAgent {}".format(self.playerId)
