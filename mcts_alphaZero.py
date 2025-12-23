# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search implementation following AlphaGo Zero methodology
Uses policy-value network for tree guidance and leaf evaluation

@author: Kevin Chen
"""

import numpy as np
import copy


def computeSoftmax(x):
    probabilities = np.exp(x - np.max(x))
    probabilities /= np.sum(probabilities)
    return probabilities


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


class MonteCarloTreeSearch(object):
    """AlphaZero-style Monte Carlo Tree Search implementation"""

    def __init__(self, policyValueFn, explorationWeight=5, numSimulations=10000):
        """
        policyValueFn: function that returns (action_probs, value) for a state
        explorationWeight: UCB exploration constant (c_puct)
        numSimulations: number of MCTS iterations per move
        """
        self._rootNode = SearchNode(None, 1.0)
        self._evaluator = policyValueFn
        self._explorationWeight = explorationWeight
        self._numSimulations = numSimulations

    def _runSimulation(self, gameState):
        """
        Execute one simulation from root to leaf, evaluate leaf, backpropagate.
        gameState is modified in-place; caller must provide a copy.
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
        Run all simulations and return action probabilities.
        temperature: controls exploration in action selection
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
        Move root to child corresponding to lastAction.
        Preserves subtree knowledge.
        """
        if lastAction in self._rootNode._childNodes:
            self._rootNode = self._rootNode._childNodes[lastAction]
            self._rootNode._parentNode = None
        else:
            self._rootNode = SearchNode(None, 1.0)

    def __str__(self):
        return "MonteCarloTreeSearch"


class TreeSearchAgent(object):
    """AI player using MCTS with neural network guidance"""

    def __init__(self, policyValueFn,
                 explorationWeight=5, numSimulations=2000, selfPlayMode=0):
        self.searchTree = MonteCarloTreeSearch(policyValueFn, explorationWeight, numSimulations)
        self._selfPlayMode = selfPlayMode

    def assignPlayerId(self, playerId):
        self.playerId = playerId

    def resetState(self):
        self.searchTree.advanceTree(-1)

    def selectMove(self, gameState, temperature=1e-3, returnProb=0):
        validMoves = gameState.openPositions
        moveProbabilities = np.zeros(gameState.cols * gameState.rows)
        if len(validMoves) > 0:
            actions, probs = self.searchTree.computeMoveDistribution(gameState, temperature)
            moveProbabilities[list(actions)] = probs
            if self._selfPlayMode:
                # Add Dirichlet noise for exploration during self-play
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
            print("WARNING: no valid moves available")

    def __str__(self):
        return "TreeSearchAgent {}".format(self.playerId)
