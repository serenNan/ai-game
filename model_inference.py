# -*- coding: utf-8 -*-
"""
Policy-Value Network implementation using pure NumPy
Enables inference without deep learning framework dependencies

@author: Kevin Chen
"""

from __future__ import print_function
import numpy as np


def computeSoftmax(x):
    probabilities = np.exp(x - np.max(x))
    probabilities /= np.sum(probabilities)
    return probabilities


def applyRelu(tensor):
    return np.maximum(tensor, 0)


def convolutionForward(inputTensor, weights, bias, stride=1, padding=1):
    numFilters, filterDepth, filterHeight, filterWidth = weights.shape
    # Theano conv2d rotates filters 180 degrees
    weights = weights[:, :, ::-1, ::-1]
    batchSize, inputDepth, inputHeight, inputWidth = inputTensor.shape
    outputHeight = (inputHeight - filterHeight + 2 * padding) / stride + 1
    outputWidth = (inputWidth - filterWidth + 2 * padding) / stride + 1
    outputHeight, outputWidth = int(outputHeight), int(outputWidth)
    colMatrix = imageToColumns(inputTensor, filterHeight, filterWidth,
                               padding=padding, stride=stride)
    weightMatrix = weights.reshape(numFilters, -1)
    output = (np.dot(weightMatrix, colMatrix).T + bias).T
    output = output.reshape(numFilters, outputHeight, outputWidth, batchSize)
    output = output.transpose(3, 0, 1, 2)
    return output


def fullyConnectedForward(inputTensor, weights, bias):
    return np.dot(inputTensor, weights) + bias


def computeIm2ColIndices(tensorShape, filterHeight,
                         filterWidth, padding=1, stride=1):
    batchSize, channels, height, width = tensorShape
    assert (height + 2 * padding - filterHeight) % stride == 0
    assert (width + 2 * padding - filterWidth) % stride == 0
    outputHeight = int((height + 2 * padding - filterHeight) / stride + 1)
    outputWidth = int((width + 2 * padding - filterWidth) / stride + 1)

    rowIdx0 = np.repeat(np.arange(filterHeight), filterWidth)
    rowIdx0 = np.tile(rowIdx0, channels)
    rowIdx1 = stride * np.repeat(np.arange(outputHeight), outputWidth)
    colIdx0 = np.tile(np.arange(filterWidth), filterHeight * channels)
    colIdx1 = stride * np.tile(np.arange(outputWidth), outputHeight)
    rowIndices = rowIdx0.reshape(-1, 1) + rowIdx1.reshape(1, -1)
    colIndices = colIdx0.reshape(-1, 1) + colIdx1.reshape(1, -1)

    channelIndices = np.repeat(np.arange(channels), filterHeight * filterWidth).reshape(-1, 1)

    return (channelIndices.astype(int), rowIndices.astype(int), colIndices.astype(int))


def imageToColumns(inputTensor, filterHeight, filterWidth, padding=1, stride=1):
    """Convert image patches to column matrix for efficient convolution"""
    p = padding
    paddedInput = np.pad(inputTensor, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    channelIdx, rowIdx, colIdx = computeIm2ColIndices(inputTensor.shape, filterHeight,
                                                      filterWidth, padding, stride)

    columns = paddedInput[:, channelIdx, rowIdx, colIdx]
    channels = inputTensor.shape[1]
    columns = columns.transpose(1, 2, 0).reshape(filterHeight * filterWidth * channels, -1)
    return columns


class NumpyNetworkEvaluator():
    """Policy-value network using pure NumPy for inference"""

    def __init__(self, boardCols, boardRows, networkParams):
        self.boardCols = boardCols
        self.boardRows = boardRows
        self.weights = networkParams

    def evaluatePosition(self, gameState):
        """
        Evaluate board position
        Returns: (action, probability) tuples and position value
        """
        validMoves = gameState.openPositions
        currentState = gameState.getStateArray()

        x = currentState.reshape(-1, 4, self.boardCols, self.boardRows)
        # Three convolutional layers with ReLU
        for i in [0, 2, 4]:
            x = applyRelu(convolutionForward(x, self.weights[i], self.weights[i + 1]))
        # Policy head
        policyX = applyRelu(convolutionForward(x, self.weights[6], self.weights[7], padding=0))
        policyX = fullyConnectedForward(policyX.flatten(), self.weights[8], self.weights[9])
        actionProbs = computeSoftmax(policyX)
        # Value head
        valueX = applyRelu(convolutionForward(x, self.weights[10],
                                              self.weights[11], padding=0))
        valueX = applyRelu(fullyConnectedForward(valueX.flatten(), self.weights[12], self.weights[13]))
        positionValue = np.tanh(fullyConnectedForward(valueX, self.weights[14], self.weights[15]))[0]
        actionProbs = zip(validMoves, actionProbs.flatten()[validMoves])
        return actionProbs, positionValue
