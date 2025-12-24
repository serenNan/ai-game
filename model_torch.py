"""
使用 PyTorch 实现的策略价值网络
为 AlphaZero 的蒙特卡洛树搜索提供神经网络引导
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def adjustLearningRate(optimizer, newLr):
    """更新优化器的学习率"""
    for paramGroup in optimizer.param_groups:
        paramGroup['lr'] = newLr


class ConvolutionalNetwork(nn.Module):
    """用于策略和价值估计的神经网络架构"""

    def __init__(self, boardCols, boardRows):
        super(ConvolutionalNetwork, self).__init__()

        self.boardCols = boardCols
        self.boardRows = boardRows
        # 共享卷积层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 策略头层
        self.policyConv = nn.Conv2d(128, 4, kernel_size=1)
        self.policyFc = nn.Linear(4 * boardCols * boardRows,
                                  boardCols * boardRows)
        # 价值头层
        self.valueConv = nn.Conv2d(128, 2, kernel_size=1)
        self.valueFc1 = nn.Linear(2 * boardCols * boardRows, 64)
        self.valueFc2 = nn.Linear(64, 1)

    def forward(self, inputState):
        """前向传播"""
        # 共享层
        x = F.relu(self.conv1(inputState))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 策略头
        policyOut = F.relu(self.policyConv(x))
        policyOut = policyOut.view(-1, 4 * self.boardCols * self.boardRows)
        policyOut = F.log_softmax(self.policyFc(policyOut), dim=1)
        # 价值头
        valueOut = F.relu(self.valueConv(x))
        valueOut = valueOut.view(-1, 2 * self.boardCols * self.boardRows)
        valueOut = F.relu(self.valueFc1(valueOut))
        valueOut = torch.tanh(self.valueFc2(valueOut))
        return policyOut, valueOut


class NeuralNetworkEvaluator():
    """神经网络训练和推理的封装类"""

    def __init__(self, boardCols, boardRows,
                 modelPath=None, useGpu=False):
        self.useGpu = useGpu
        self.boardCols = boardCols
        self.boardRows = boardRows
        self.l2Regularization = 1e-4
        # 初始化网络
        if self.useGpu:
            self.network = ConvolutionalNetwork(boardCols, boardRows).cuda()
        else:
            self.network = ConvolutionalNetwork(boardCols, boardRows)
        self.optimizer = optim.Adam(self.network.parameters(),
                                    weight_decay=self.l2Regularization)

        if modelPath:
            savedParams = torch.load(modelPath)
            self.network.load_state_dict(savedParams)

    def batchEvaluate(self, stateBatch):
        """
        批量评估状态
        返回: 动作概率和状态价值
        """
        if self.useGpu:
            stateBatch = Variable(torch.FloatTensor(stateBatch).cuda())
            logProbs, values = self.network(stateBatch)
            actionProbs = np.exp(logProbs.data.cpu().numpy())
            return actionProbs, values.data.cpu().numpy()
        else:
            stateBatch = Variable(torch.FloatTensor(stateBatch))
            logProbs, values = self.network(stateBatch)
            actionProbs = np.exp(logProbs.data.numpy())
            return actionProbs, values.data.numpy()

    def evaluatePosition(self, gameState):
        """
        评估单个棋盘局面
        返回: (动作, 概率) 元组和局面价值
        """
        validMoves = gameState.openPositions
        currentState = np.ascontiguousarray(gameState.getStateArray().reshape(
            -1, 4, self.boardCols, self.boardRows))
        if self.useGpu:
            logProbs, value = self.network(
                Variable(torch.from_numpy(currentState)).cuda().float())
            actionProbs = np.exp(logProbs.data.cpu().numpy().flatten())
            value = value.data.cpu().numpy()[0][0]
        else:
            logProbs, value = self.network(
                Variable(torch.from_numpy(currentState)).float())
            actionProbs = np.exp(logProbs.data.numpy().flatten())
            value = value.data.numpy()[0][0]
        actionProbs = zip(validMoves, actionProbs[validMoves])
        return actionProbs, value

    def trainOnBatch(self, stateBatch, targetProbs, targetOutcomes, learningRate):
        """执行一次训练步骤"""
        if self.useGpu:
            stateBatch = Variable(torch.FloatTensor(stateBatch).cuda())
            targetProbs = Variable(torch.FloatTensor(targetProbs).cuda())
            targetOutcomes = Variable(torch.FloatTensor(targetOutcomes).cuda())
        else:
            stateBatch = Variable(torch.FloatTensor(stateBatch))
            targetProbs = Variable(torch.FloatTensor(targetProbs))
            targetOutcomes = Variable(torch.FloatTensor(targetOutcomes))

        self.optimizer.zero_grad()
        adjustLearningRate(self.optimizer, learningRate)

        logProbs, values = self.network(stateBatch)
        # 损失函数: (z - v)^2 - pi^T * log(p) + c||theta||^2
        valueLoss = F.mse_loss(values.view(-1), targetOutcomes)
        policyLoss = -torch.mean(torch.sum(targetProbs * logProbs, 1))
        totalLoss = valueLoss + policyLoss
        totalLoss.backward()
        self.optimizer.step()
        # 计算熵用于监控
        entropy = -torch.mean(
            torch.sum(torch.exp(logProbs) * logProbs, 1)
        )
        return totalLoss.item(), entropy.item()

    def getNetworkParams(self):
        """获取网络参数"""
        return self.network.state_dict()

    def saveCheckpoint(self, filePath):
        """保存模型参数到文件"""
        params = self.getNetworkParams()
        torch.save(params, filePath)
