"""
五子棋 AlphaZero 训练流水线
实现自我对弈、数据增强和迭代策略改进
"""

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from board import GameState, GameController
from random_search import PureSearchAgent
from neural_search import TreeSearchAgent
from model_torch import NeuralNetworkEvaluator


class TrainingManager():
    """训练管理器，负责整个训练流程"""

    def __init__(self, modelPath=None):
        # 棋盘配置
        self.boardCols = 6
        self.boardRows = 6
        self.winLength = 4
        self.gameState = GameState(width=self.boardCols,
                                   height=self.boardRows,
                                   n_in_row=self.winLength)
        self.controller = GameController(self.gameState)
        # 训练超参数
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
        # 基准对手强度
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
        通过旋转和翻转进行数据增强
        gameData: [(状态, 落子概率, 结果), ...]
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
                # 水平翻转
                flippedState = np.array([np.fliplr(s) for s in rotatedState])
                flippedProbs = np.fliplr(rotatedProbs)
                augmentedData.append((flippedState,
                                      np.flipud(flippedProbs).flatten(),
                                      outcome))
        return augmentedData

    def generateSelfPlayData(self, numGames=1):
        """通过自我对弈生成训练数据"""
        for _ in range(numGames):
            _, gameData = self.controller.runSelfPlay(self.trainAgent,
                                                      temperature=self.explorationTemp)
            gameData = list(gameData)[:]
            self.episodeLength = len(gameData)
            augmentedData = self.augmentData(gameData)
            self.replayBuffer.extend(augmentedData)

    def updatePolicy(self):
        """在采样数据上训练神经网络"""
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
        # 自适应学习率
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
        print(("KL散度:{:.5f},"
               "学习率倍数:{:.3f},"
               "损失:{},"
               "熵:{},"
               "前解释方差:{:.3f},"
               "后解释方差:{:.3f}"
               ).format(klDivergence,
                        self.learningRateScale,
                        loss,
                        entropy,
                        explainedVarPrev,
                        explainedVarNew))
        return loss, entropy

    def evaluatePolicy(self, numGames=10):
        """
        与基准纯 MCTS 对战评估当前策略
        用于监控训练进度
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
        print("基准模拟次数:{}, 胜: {}, 负: {}, 平:{}".format(
            self.baselineSimulations,
            winCounts[1], winCounts[2], winCounts[-1]))
        return winRate

    def runTraining(self):
        """执行完整的训练循环"""
        try:
            for batchIdx in range(self.totalBatches):
                self.generateSelfPlayData(self.gamesPerBatch)
                print("批次:{}, 对局步数:{}".format(
                    batchIdx + 1, self.episodeLength))
                if len(self.replayBuffer) > self.miniBatchSize:
                    self.updatePolicy()
                if (batchIdx + 1) % self.evaluationInterval == 0:
                    print("当前批次: {}".format(batchIdx + 1))
                    winRate = self.evaluatePolicy()
                    self.neuralNetwork.saveCheckpoint('./current_policy.model')
                    if winRate > self.bestWinRate:
                        print("发现新的最佳策略!")
                        self.bestWinRate = winRate
                        self.neuralNetwork.saveCheckpoint('./best_policy.model')
                        if (self.bestWinRate == 1.0 and
                                self.baselineSimulations < 5000):
                            self.baselineSimulations += 1000
                            self.bestWinRate = 0.0
        except KeyboardInterrupt:
            print('\n\r训练已中断')


if __name__ == '__main__':
    manager = TrainingManager()
    manager.runTraining()
