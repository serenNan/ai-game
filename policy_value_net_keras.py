# -*- coding: utf-8 -*-
"""
Policy-Value Network implementation using Keras
Tested with Keras 2.0.5 and tensorflow-gpu 1.2.1

@author: Kevin Chen
"""

from __future__ import print_function

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K

from keras.utils import np_utils

import numpy as np
import pickle


class NeuralNetworkEvaluator():
    """Policy-value network using Keras"""

    def __init__(self, boardCols, boardRows, modelPath=None):
        self.boardCols = boardCols
        self.boardRows = boardRows
        self.l2Regularization = 1e-4
        self._buildNetwork()
        self._setupTraining()

        if modelPath:
            savedParams = pickle.load(open(modelPath, 'rb'))
            self.model.set_weights(savedParams)

    def _buildNetwork(self):
        """Construct the neural network architecture"""
        inputLayer = network = Input((4, self.boardCols, self.boardRows))

        # Shared convolutional layers
        network = Conv2D(filters=32, kernel_size=(3, 3), padding="same",
                         data_format="channels_first", activation="relu",
                         kernel_regularizer=l2(self.l2Regularization))(network)
        network = Conv2D(filters=64, kernel_size=(3, 3), padding="same",
                         data_format="channels_first", activation="relu",
                         kernel_regularizer=l2(self.l2Regularization))(network)
        network = Conv2D(filters=128, kernel_size=(3, 3), padding="same",
                         data_format="channels_first", activation="relu",
                         kernel_regularizer=l2(self.l2Regularization))(network)

        # Policy head
        policyNetwork = Conv2D(filters=4, kernel_size=(1, 1),
                               data_format="channels_first", activation="relu",
                               kernel_regularizer=l2(self.l2Regularization))(network)
        policyNetwork = Flatten()(policyNetwork)
        self.policyOutput = Dense(self.boardCols * self.boardRows,
                                  activation="softmax",
                                  kernel_regularizer=l2(self.l2Regularization))(policyNetwork)

        # Value head
        valueNetwork = Conv2D(filters=2, kernel_size=(1, 1),
                              data_format="channels_first", activation="relu",
                              kernel_regularizer=l2(self.l2Regularization))(network)
        valueNetwork = Flatten()(valueNetwork)
        valueNetwork = Dense(64, kernel_regularizer=l2(self.l2Regularization))(valueNetwork)
        self.valueOutput = Dense(1, activation="tanh",
                                 kernel_regularizer=l2(self.l2Regularization))(valueNetwork)

        self.model = Model(inputLayer, [self.policyOutput, self.valueOutput])

        def batchPredict(stateInput):
            stateArray = np.array(stateInput)
            return self.model.predict_on_batch(stateArray)

        self.batchEvaluate = batchPredict

    def evaluatePosition(self, gameState):
        """
        Evaluate board position
        Returns: (action, probability) tuples and position value
        """
        validMoves = gameState.openPositions
        currentState = gameState.getStateArray()
        actionProbs, value = self.batchEvaluate(
            currentState.reshape(-1, 4, self.boardCols, self.boardRows))
        actionProbs = zip(validMoves, actionProbs.flatten()[validMoves])
        return actionProbs, value[0][0]

    def _setupTraining(self):
        """Configure training operations"""
        optimizer = Adam()
        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.compile(optimizer=optimizer, loss=losses)

        def computeEntropy(probs):
            return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        def executeTrainStep(stateInput, targetProbs, targetOutcomes, learningRate):
            stateArray = np.array(stateInput)
            probsArray = np.array(targetProbs)
            outcomesArray = np.array(targetOutcomes)
            loss = self.model.evaluate(stateArray, [probsArray, outcomesArray],
                                       batch_size=len(stateInput), verbose=0)
            actionProbs, _ = self.model.predict_on_batch(stateArray)
            entropy = computeEntropy(actionProbs)
            K.set_value(self.model.optimizer.lr, learningRate)
            self.model.fit(stateArray, [probsArray, outcomesArray],
                           batch_size=len(stateInput), verbose=0)
            return loss[0], entropy

        self.trainOnBatch = executeTrainStep

    def getNetworkParams(self):
        return self.model.get_weights()

    def saveCheckpoint(self, filePath):
        """Save model parameters to file"""
        params = self.getNetworkParams()
        pickle.dump(params, open(filePath, 'wb'), protocol=2)
