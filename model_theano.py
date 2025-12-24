"""
使用 Theano/Lasagne 实现的策略价值网络
为 AlphaZero 的蒙特卡洛树搜索提供神经网络引导
"""

from __future__ import print_function
import theano
import theano.tensor as T
import lasagne
import pickle


class NeuralNetworkEvaluator():
    """使用 Theano/Lasagne 实现的策略价值网络"""

    def __init__(self, boardCols, boardRows, modelPath=None):
        self.boardCols = boardCols
        self.boardRows = boardRows
        self.learningRate = T.scalar('learning_rate')
        self.l2Regularization = 1e-4
        self._buildNetwork()
        self._setupTraining()
        if modelPath:
            with open(modelPath, 'rb') as f:
                try:
                    savedParams = pickle.load(f)
                except (UnicodeDecodeError, KeyError):
                    f.seek(0)
                    savedParams = pickle.load(f, encoding='bytes')
            lasagne.layers.set_all_param_values(
                [self.policyLayer, self.valueLayer], savedParams)

    def _buildNetwork(self):
        """构建神经网络架构"""
        self.inputState = T.tensor4('state')
        self.targetOutcome = T.vector('winner')
        self.targetProbs = T.matrix('mcts_probs')
        network = lasagne.layers.InputLayer(
            shape=(None, 4, self.boardCols, self.boardRows),
            input_var=self.inputState)

        # 共享卷积层
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3), pad='same')
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3, 3), pad='same')
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(3, 3), pad='same')

        # 策略头
        policyNetwork = lasagne.layers.Conv2DLayer(
            network, num_filters=4, filter_size=(1, 1))
        self.policyLayer = lasagne.layers.DenseLayer(
            policyNetwork, num_units=self.boardCols * self.boardRows,
            nonlinearity=lasagne.nonlinearities.softmax)

        # 价值头
        valueNetwork = lasagne.layers.Conv2DLayer(
            network, num_filters=2, filter_size=(1, 1))
        valueNetwork = lasagne.layers.DenseLayer(valueNetwork, num_units=64)
        self.valueLayer = lasagne.layers.DenseLayer(
            valueNetwork, num_units=1,
            nonlinearity=lasagne.nonlinearities.tanh)

        # 获取输出
        self.actionProbs, self.positionValue = lasagne.layers.get_output(
            [self.policyLayer, self.valueLayer])
        self.batchEvaluate = theano.function([self.inputState],
                                             [self.actionProbs, self.positionValue],
                                             allow_input_downcast=True)

    def evaluatePosition(self, gameState):
        """
        评估棋盘局面
        返回: (动作, 概率) 元组和局面价值
        """
        validMoves = gameState.openPositions
        currentState = gameState.getStateArray()
        actionProbs, value = self.batchEvaluate(
            currentState.reshape(-1, 4, self.boardCols, self.boardRows))
        actionProbs = zip(validMoves, actionProbs.flatten()[validMoves])
        return actionProbs, value[0][0]

    def _setupTraining(self):
        """配置损失函数和优化器"""
        params = lasagne.layers.get_all_params(
            [self.policyLayer, self.valueLayer], trainable=True)
        valueLoss = lasagne.objectives.squared_error(
            self.targetOutcome, self.positionValue.flatten())
        policyLoss = lasagne.objectives.categorical_crossentropy(
            self.actionProbs, self.targetProbs)
        l2Penalty = lasagne.regularization.apply_penalty(
            params, lasagne.regularization.l2)
        self.totalLoss = self.l2Regularization * l2Penalty + lasagne.objectives.aggregate(
            valueLoss + policyLoss, mode='mean')
        # 用于监控的熵
        self.policyEntropy = -T.mean(T.sum(
            self.actionProbs * T.log(self.actionProbs + 1e-10), axis=1))
        # 优化器
        updates = lasagne.updates.adam(self.totalLoss, params,
                                       learning_rate=self.learningRate)
        self.trainOnBatch = theano.function(
            [self.inputState, self.targetProbs, self.targetOutcome, self.learningRate],
            [self.totalLoss, self.policyEntropy],
            updates=updates,
            allow_input_downcast=True)

    def getNetworkParams(self):
        """获取网络参数"""
        return lasagne.layers.get_all_param_values(
            [self.policyLayer, self.valueLayer])

    def saveCheckpoint(self, filePath):
        """保存模型参数到文件"""
        params = self.getNetworkParams()
        pickle.dump(params, open(filePath, 'wb'), protocol=2)
