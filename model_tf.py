"""
使用 TensorFlow 实现的策略价值网络
已在 TensorFlow 1.4 和 1.5 版本测试通过
"""

import numpy as np
import tensorflow as tf


class NeuralNetworkEvaluator():
    def __init__(self, boardCols, boardRows, modelPath=None):
        self.boardCols = boardCols
        self.boardRows = boardRows

        # Input placeholders
        self.inputStates = tf.placeholder(
            tf.float32, shape=[None, 4, boardRows, boardCols])
        self.inputStateTrans = tf.transpose(self.inputStates, [0, 2, 3, 1])

        # Shared convolutional layers
        self.conv1 = tf.layers.conv2d(inputs=self.inputStateTrans,
                                      filters=32, kernel_size=[3, 3],
                                      padding="same", data_format="channels_last",
                                      activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64,
                                      kernel_size=[3, 3], padding="same",
                                      data_format="channels_last",
                                      activation=tf.nn.relu)
        self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=128,
                                      kernel_size=[3, 3], padding="same",
                                      data_format="channels_last",
                                      activation=tf.nn.relu)

        # Policy head
        self.policyConv = tf.layers.conv2d(inputs=self.conv3, filters=4,
                                           kernel_size=[1, 1], padding="same",
                                           data_format="channels_last",
                                           activation=tf.nn.relu)
        self.policyFlat = tf.reshape(
            self.policyConv, [-1, 4 * boardRows * boardCols])
        self.policyOutput = tf.layers.dense(inputs=self.policyFlat,
                                            units=boardRows * boardCols,
                                            activation=tf.nn.log_softmax)

        # Value head
        self.valueConv = tf.layers.conv2d(inputs=self.conv3, filters=2,
                                          kernel_size=[1, 1],
                                          padding="same",
                                          data_format="channels_last",
                                          activation=tf.nn.relu)
        self.valueFlat = tf.reshape(
            self.valueConv, [-1, 2 * boardRows * boardCols])
        self.valueFc1 = tf.layers.dense(inputs=self.valueFlat,
                                        units=64, activation=tf.nn.relu)
        self.valueOutput = tf.layers.dense(inputs=self.valueFc1,
                                           units=1, activation=tf.nn.tanh)

        # Loss function components
        self.targetOutcomes = tf.placeholder(tf.float32, shape=[None, 1])
        self.valueLoss = tf.losses.mean_squared_error(self.targetOutcomes,
                                                      self.valueOutput)
        self.targetProbs = tf.placeholder(
            tf.float32, shape=[None, boardRows * boardCols])
        self.policyLoss = tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.multiply(self.targetProbs, self.policyOutput), 1)))

        # L2 regularization
        l2Weight = 1e-4
        trainableVars = tf.trainable_variables()
        l2Penalty = l2Weight * tf.add_n(
            [tf.nn.l2_loss(v) for v in trainableVars if 'bias' not in v.name.lower()])
        self.totalLoss = self.valueLoss + self.policyLoss + l2Penalty

        # Optimizer
        self.learningRate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learningRate).minimize(self.totalLoss)

        # Session setup
        self.session = tf.Session()

        # Policy entropy for monitoring
        self.policyEntropy = tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.exp(self.policyOutput) * self.policyOutput, 1)))

        # Initialize variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # Model persistence
        self.saver = tf.train.Saver()
        if modelPath is not None:
            self.loadCheckpoint(modelPath)

    def batchEvaluate(self, stateBatch):
        """Evaluate a batch of states"""
        logProbs, values = self.session.run(
            [self.policyOutput, self.valueOutput],
            feed_dict={self.inputStates: stateBatch})
        actionProbs = np.exp(logProbs)
        return actionProbs, values

    def evaluatePosition(self, gameState):
        """Evaluate single board position"""
        validMoves = gameState.openPositions
        currentState = np.ascontiguousarray(gameState.getStateArray().reshape(
            -1, 4, self.boardCols, self.boardRows))
        actionProbs, value = self.batchEvaluate(currentState)
        actionProbs = zip(validMoves, actionProbs[0][validMoves])
        return actionProbs, value

    def trainOnBatch(self, stateBatch, targetProbs, targetOutcomes, learningRate):
        """Execute one training step"""
        targetOutcomes = np.reshape(targetOutcomes, (-1, 1))
        loss, entropy, _ = self.session.run(
            [self.totalLoss, self.policyEntropy, self.optimizer],
            feed_dict={self.inputStates: stateBatch,
                       self.targetProbs: targetProbs,
                       self.targetOutcomes: targetOutcomes,
                       self.learningRate: learningRate})
        return loss, entropy

    def saveCheckpoint(self, filePath):
        self.saver.save(self.session, filePath)

    def loadCheckpoint(self, filePath):
        self.saver.restore(self.session, filePath)
