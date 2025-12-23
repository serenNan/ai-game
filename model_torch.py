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
    """Update optimizer learning rate"""
    for paramGroup in optimizer.param_groups:
        paramGroup['lr'] = newLr


class ConvolutionalNetwork(nn.Module):
    """Neural network architecture for policy and value estimation"""

    def __init__(self, boardCols, boardRows):
        super(ConvolutionalNetwork, self).__init__()

        self.boardCols = boardCols
        self.boardRows = boardRows
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Policy head layers
        self.policyConv = nn.Conv2d(128, 4, kernel_size=1)
        self.policyFc = nn.Linear(4 * boardCols * boardRows,
                                  boardCols * boardRows)
        # Value head layers
        self.valueConv = nn.Conv2d(128, 2, kernel_size=1)
        self.valueFc1 = nn.Linear(2 * boardCols * boardRows, 64)
        self.valueFc2 = nn.Linear(64, 1)

    def forward(self, inputState):
        # Shared layers
        x = F.relu(self.conv1(inputState))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Policy head
        policyOut = F.relu(self.policyConv(x))
        policyOut = policyOut.view(-1, 4 * self.boardCols * self.boardRows)
        policyOut = F.log_softmax(self.policyFc(policyOut))
        # Value head
        valueOut = F.relu(self.valueConv(x))
        valueOut = valueOut.view(-1, 2 * self.boardCols * self.boardRows)
        valueOut = F.relu(self.valueFc1(valueOut))
        valueOut = F.tanh(self.valueFc2(valueOut))
        return policyOut, valueOut


class NeuralNetworkEvaluator():
    """Wrapper class for neural network training and inference"""

    def __init__(self, boardCols, boardRows,
                 modelPath=None, useGpu=False):
        self.useGpu = useGpu
        self.boardCols = boardCols
        self.boardRows = boardRows
        self.l2Regularization = 1e-4
        # Initialize network
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
        Evaluate a batch of states
        Returns: action probabilities and state values
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
        Evaluate single board position
        Returns: (action, probability) tuples and position value
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
        """Execute one training step"""
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
        # Loss: (z - v)^2 - pi^T * log(p) + c||theta||^2
        valueLoss = F.mse_loss(values.view(-1), targetOutcomes)
        policyLoss = -torch.mean(torch.sum(targetProbs * logProbs, 1))
        totalLoss = valueLoss + policyLoss
        totalLoss.backward()
        self.optimizer.step()
        # Compute entropy for monitoring
        entropy = -torch.mean(
            torch.sum(torch.exp(logProbs) * logProbs, 1)
        )
        return totalLoss.data[0], entropy.data[0]
        # For PyTorch >= 0.5 use:
        # return totalLoss.item(), entropy.item()

    def getNetworkParams(self):
        return self.network.state_dict()

    def saveCheckpoint(self, filePath):
        """Save model parameters to file"""
        params = self.getNetworkParams()
        torch.save(params, filePath)
