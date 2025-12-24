"""
使用 TensorFlow 实现的策略价值网络
为 AlphaZero 的蒙特卡洛树搜索提供神经网络引导
"""

import numpy as np
import tensorflow as tf


class NeuralNetworkEvaluator():
    """使用 TensorFlow 实现的策略价值网络"""

    def __init__(self, boardCols, boardRows, modelPath=None):
        self.boardCols = boardCols
        self.boardRows = boardRows

        # 输入占位符
        self.inputStates = tf.placeholder(
            tf.float32, shape=[None, 4, boardRows, boardCols])
        self.inputStateTrans = tf.transpose(self.inputStates, [0, 2, 3, 1])

        # 共享卷积层
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

        # 策略头
        self.policyConv = tf.layers.conv2d(inputs=self.conv3, filters=4,
                                           kernel_size=[1, 1], padding="same",
                                           data_format="channels_last",
                                           activation=tf.nn.relu)
        self.policyFlat = tf.reshape(
            self.policyConv, [-1, 4 * boardRows * boardCols])
        self.policyOutput = tf.layers.dense(inputs=self.policyFlat,
                                            units=boardRows * boardCols,
                                            activation=tf.nn.log_softmax)

        # 价值头
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

        # 损失函数组件
        self.targetOutcomes = tf.placeholder(tf.float32, shape=[None, 1])
        self.valueLoss = tf.losses.mean_squared_error(self.targetOutcomes,
                                                      self.valueOutput)
        self.targetProbs = tf.placeholder(
            tf.float32, shape=[None, boardRows * boardCols])
        self.policyLoss = tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.multiply(self.targetProbs, self.policyOutput), 1)))

        # L2 正则化
        l2Weight = 1e-4
        trainableVars = tf.trainable_variables()
        l2Penalty = l2Weight * tf.add_n(
            [tf.nn.l2_loss(v) for v in trainableVars if 'bias' not in v.name.lower()])
        self.totalLoss = self.valueLoss + self.policyLoss + l2Penalty

        # 优化器
        self.learningRate = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learningRate).minimize(self.totalLoss)

        # 会话设置
        self.session = tf.Session()

        # 用于监控的策略熵
        self.policyEntropy = tf.negative(tf.reduce_mean(
            tf.reduce_sum(tf.exp(self.policyOutput) * self.policyOutput, 1)))

        # 初始化变量
        init = tf.global_variables_initializer()
        self.session.run(init)

        # 模型持久化
        self.saver = tf.train.Saver()
        if modelPath is not None:
            self.loadCheckpoint(modelPath)

    def batchEvaluate(self, stateBatch):
        """批量评估状态"""
        logProbs, values = self.session.run(
            [self.policyOutput, self.valueOutput],
            feed_dict={self.inputStates: stateBatch})
        actionProbs = np.exp(logProbs)
        return actionProbs, values

    def evaluatePosition(self, gameState):
        """评估单个棋盘局面"""
        validMoves = gameState.openPositions
        currentState = np.ascontiguousarray(gameState.getStateArray().reshape(
            -1, 4, self.boardCols, self.boardRows))
        actionProbs, value = self.batchEvaluate(currentState)
        actionProbs = zip(validMoves, actionProbs[0][validMoves])
        return actionProbs, value

    def trainOnBatch(self, stateBatch, targetProbs, targetOutcomes, learningRate):
        """执行一次训练步骤"""
        targetOutcomes = np.reshape(targetOutcomes, (-1, 1))
        loss, entropy, _ = self.session.run(
            [self.totalLoss, self.policyEntropy, self.optimizer],
            feed_dict={self.inputStates: stateBatch,
                       self.targetProbs: targetProbs,
                       self.targetOutcomes: targetOutcomes,
                       self.learningRate: learningRate})
        return loss, entropy

    def saveCheckpoint(self, filePath):
        """保存模型到文件"""
        self.saver.save(self.session, filePath)

    def loadCheckpoint(self, filePath):
        """从文件加载模型"""
        self.saver.restore(self.session, filePath)
