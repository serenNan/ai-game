# -*- coding: utf-8 -*-
"""
AlphaZero training pipeline for Gomoku
Implements self-play, data augmentation, and iterative policy improvement

@author: Kevin Chen
"""

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from game import GameState, GameController
from mcts_pure import PureSearchAgent
from mcts_alphaZero import TreeSearchAgent
from policy_value_net import NeuralNetworkEvaluator  # Theano and Lasagne
# from policy_value_net_pytorch import NeuralNetworkEvaluator  # Pytorch
# from policy_value_net_tensorflow import NeuralNetworkEvaluator  # Tensorflow
# from policy_value_net_keras import NeuralNetworkEvaluator  # Keras


class TrainingManager():
    def __init__(self, modelPath=None):
        # Board configuration
        self.boardCols = 6
        self.boardRows = 6
        self.winLength = 4
        self.gameState = GameState(width=self.boardCols,
                                   height=self.boardRows,
                                   n_in_row=self.winLength)
        self.controller = GameController(self.gameState)
        # Training hyperparameters
        self.baseLearningRate = 2e-3
        self.learningRateScale = 1.0
        self.explorationTemp = 1.0
        self.numSimulations = 400
        self.explorationWeight = 5
        self.replayBufferSize = 10000
        self.miniBatchSize = 512
        self.replayBuffer = deque(maxlen=self.replayBufferSize)
        self.gamesPerBatch = 1
        self.trainingEpochs = 5
        self.klTarget = 0.02
        self.evaluationInterval = 50
        self.totalBatches = 1500
        self.bestWinRate = 0.0
        # Baseline opponent strength
        self.baselineSimulations = 1000
        if modelPath:
            self.neuralNetwork = NeuralNetworkEvaluator(self.boardCols,
                                                        self.boardRows,
                                                        modelPath=modelPath)
        else:
            self.neuralNetwork = NeuralNetworkEvaluator(self.boardCols,
                                                        self.boardRows)
        self.trainAgent = TreeSearchAgent(self.neuralNetwork.evaluatePosition,
                                          explorationWeight=self.explorationWeight,
                                          numSimulations=self.numSimulations,
                                          selfPlayMode=1)

    def augmentData(self, gameData):
        """
        Augment dataset with rotations and reflections.
        gameData: [(state, moveProbs, outcome), ...]
        """
        augmentedData = []
        for state, moveProbs, outcome in gameData:
            for rotation in [1, 2, 3, 4]:
                rotatedState = np.array([np.rot90(s, rotation) for s in state])
                rotatedProbs = np.rot90(np.flipud(
                    moveProbs.reshape(self.boardRows, self.boardCols)), rotation)
                augmentedData.append((rotatedState,
                                      np.flipud(rotatedProbs).flatten(),
                                      outcome))
                # Horizontal flip
                flippedState = np.array([np.fliplr(s) for s in rotatedState])
                flippedProbs = np.fliplr(rotatedProbs)
                augmentedData.append((flippedState,
                                      np.flipud(flippedProbs).flatten(),
                                      outcome))
        return augmentedData

    def generateSelfPlayData(self, numGames=1):
        """Generate training data through self-play"""
        for _ in range(numGames):
            victor, gameData = self.controller.runSelfPlay(self.trainAgent,
                                                           temperature=self.explorationTemp)
            gameData = list(gameData)[:]
            self.episodeLength = len(gameData)
            augmentedData = self.augmentData(gameData)
            self.replayBuffer.extend(augmentedData)

    def updatePolicy(self):
        """Train the neural network on sampled data"""
        miniBatch = random.sample(self.replayBuffer, self.miniBatchSize)
        stateBatch = [sample[0] for sample in miniBatch]
        probsBatch = [sample[1] for sample in miniBatch]
        outcomeBatch = [sample[2] for sample in miniBatch]
        prevProbs, prevValues = self.neuralNetwork.batchEvaluate(stateBatch)
        for _ in range(self.trainingEpochs):
            loss, entropy = self.neuralNetwork.trainOnBatch(
                stateBatch,
                probsBatch,
                outcomeBatch,
                self.baseLearningRate * self.learningRateScale)
            newProbs, newValues = self.neuralNetwork.batchEvaluate(stateBatch)
            klDivergence = np.mean(np.sum(prevProbs * (
                np.log(prevProbs + 1e-10) - np.log(newProbs + 1e-10)),
                axis=1))
            if klDivergence > self.klTarget * 4:
                break
        # Adaptive learning rate
        if klDivergence > self.klTarget * 2 and self.learningRateScale > 0.1:
            self.learningRateScale /= 1.5
        elif klDivergence < self.klTarget / 2 and self.learningRateScale < 10:
            self.learningRateScale *= 1.5

        explainedVarPrev = (1 -
                           np.var(np.array(outcomeBatch) - prevValues.flatten()) /
                           np.var(np.array(outcomeBatch)))
        explainedVarNew = (1 -
                          np.var(np.array(outcomeBatch) - newValues.flatten()) /
                          np.var(np.array(outcomeBatch)))
        print(("kl:{:.5f},"
               "lr_scale:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_prev:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(klDivergence,
                        self.learningRateScale,
                        loss,
                        entropy,
                        explainedVarPrev,
                        explainedVarNew))
        return loss, entropy

    def evaluatePolicy(self, numGames=10):
        """
        Evaluate current policy against baseline pure MCTS.
        Used to monitor training progress.
        """
        currentAgent = TreeSearchAgent(self.neuralNetwork.evaluatePosition,
                                       explorationWeight=self.explorationWeight,
                                       numSimulations=self.numSimulations)
        baselineAgent = PureSearchAgent(explorationWeight=5,
                                        numSimulations=self.baselineSimulations)
        winCounts = defaultdict(int)
        for gameIdx in range(numGames):
            victor = self.controller.runMatch(currentAgent,
                                              baselineAgent,
                                              firstPlayer=gameIdx % 2,
                                              displayBoard=0)
            winCounts[victor] += 1
        winRate = 1.0 * (winCounts[1] + 0.5 * winCounts[-1]) / numGames
        print("baseline_simulations:{}, wins: {}, losses: {}, draws:{}".format(
            self.baselineSimulations,
            winCounts[1], winCounts[2], winCounts[-1]))
        return winRate

    def runTraining(self):
        """Execute the full training loop"""
        try:
            for batchIdx in range(self.totalBatches):
                self.generateSelfPlayData(self.gamesPerBatch)
                print("batch:{}, episode_length:{}".format(
                    batchIdx + 1, self.episodeLength))
                if len(self.replayBuffer) > self.miniBatchSize:
                    loss, entropy = self.updatePolicy()
                if (batchIdx + 1) % self.evaluationInterval == 0:
                    print("current batch: {}".format(batchIdx + 1))
                    winRate = self.evaluatePolicy()
                    self.neuralNetwork.saveCheckpoint('./current_policy.model')
                    if winRate > self.bestWinRate:
                        print("New best policy found!")
                        self.bestWinRate = winRate
                        self.neuralNetwork.saveCheckpoint('./best_policy.model')
                        if (self.bestWinRate == 1.0 and
                                self.baselineSimulations < 5000):
                            self.baselineSimulations += 1000
                            self.bestWinRate = 0.0
        except KeyboardInterrupt:
            print('\n\rTraining interrupted')


if __name__ == '__main__':
    manager = TrainingManager()
    manager.runTraining()
