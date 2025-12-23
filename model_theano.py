"""
使用 Theano 和 Lasagne 实现的策略价值网络
默认用于训练和推理的实现
"""

from __future__ import print_function
import theano
import theano.tensor as T
import lasagne
import pickle


class NeuralNetworkEvaluator():
    """Policy-value network using Theano/Lasagne"""

    def __init__(self, boardCols, boardRows, modelPath=None):
        self.boardCols = boardCols
        self.boardRows = boardRows
        self.learningRate = T.scalar('learning_rate')
        self.l2Regularization = 1e-4
        self._buildNetwork()
        self._setupTraining()
        if modelPath:
            try:
                savedParams = pickle.load(open(modelPath, 'rb'))
            except:
                savedParams = pickle.load(open(modelPath, 'rb'),
                                          encoding='bytes')
            lasagne.layers.set_all_param_values(
                [self.policyLayer, self.valueLayer], savedParams)

    def _buildNetwork(self):
        """Construct the neural network architecture"""
        self.inputState = T.tensor4('state')
        self.targetOutcome = T.vector('winner')
        self.targetProbs = T.matrix('mcts_probs')
        network = lasagne.layers.InputLayer(
            shape=(None, 4, self.boardCols, self.boardRows),
            input_var=self.inputState)

        # Shared convolutional layers
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3), pad='same')
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3, 3), pad='same')
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(3, 3), pad='same')

        # Policy head
        policyNetwork = lasagne.layers.Conv2DLayer(
            network, num_filters=4, filter_size=(1, 1))
        self.policyLayer = lasagne.layers.DenseLayer(
            policyNetwork, num_units=self.boardCols * self.boardRows,
            nonlinearity=lasagne.nonlinearities.softmax)

        # Value head
        valueNetwork = lasagne.layers.Conv2DLayer(
            network, num_filters=2, filter_size=(1, 1))
        valueNetwork = lasagne.layers.DenseLayer(valueNetwork, num_units=64)
        self.valueLayer = lasagne.layers.DenseLayer(
            valueNetwork, num_units=1,
            nonlinearity=lasagne.nonlinearities.tanh)

        # Get outputs
        self.actionProbs, self.positionValue = lasagne.layers.get_output(
            [self.policyLayer, self.valueLayer])
        self.batchEvaluate = theano.function([self.inputState],
                                             [self.actionProbs, self.positionValue],
                                             allow_input_downcast=True)

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
        """Configure loss function and optimizer"""
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
        # Entropy for monitoring
        self.policyEntropy = -T.mean(T.sum(
            self.actionProbs * T.log(self.actionProbs + 1e-10), axis=1))
        # Optimizer
        updates = lasagne.updates.adam(self.totalLoss, params,
                                       learning_rate=self.learningRate)
        self.trainOnBatch = theano.function(
            [self.inputState, self.targetProbs, self.targetOutcome, self.learningRate],
            [self.totalLoss, self.policyEntropy],
            updates=updates,
            allow_input_downcast=True)

    def getNetworkParams(self):
        return lasagne.layers.get_all_param_values(
            [self.policyLayer, self.valueLayer])

    def saveCheckpoint(self, filePath):
        """Save model parameters to file"""
        params = self.getNetworkParams()
        pickle.dump(params, open(filePath, 'wb'), protocol=2)
