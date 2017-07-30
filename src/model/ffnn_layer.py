
import time

import numpy as np

from util.activation_functions import Activation
# from model.layer import Layer


class FFNNLayer:
    """
    A layer of perceptrons acting as the output layer

    Parameters
    ----------
    nIn: int: number of units from the previous layer (or input data)
    nOut: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    isClassifierLayer: bool:  to do classification or regression

    The bias dummy node is always first in the implicit list of nodes
    represented through weights. Note that while the bias weight must be
    updated in each layer, no error can be propagated through this node.

    Attributes
    ----------
    nIn : positive int:
        number of units from the previous layer
    nOut : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activationString : string
        the name of the activation function
    isClassifierLayer: bool
        to do classification or regression
    gradient_matrix : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, nIn, nOut, weights=None,
                 activation='softmax', isClassifierLayer=True,
                 momentumRate=0.005, regularization_rate=0.5):

        # Get activation function from string
        # Notice the functional programming paradigms of Python + Numpy
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)
        if not isClassifierLayer:
            self.activationPrime = Activation.getDerivative(self.activationString)

        self.nIn = nIn
        self.nOut = nOut

        # Adding bias
        self.input = np.ndarray((nIn+1, 1))
        self.input[0] = 1
        self.output = np.ndarray((nOut, 1))

        # weight update terms
        self.gradient_matrix = np.zeros((nOut, nIn + 1))   # gradient
        self.momentum_matrix = np.zeros((nOut, nIn + 1))   # momentum

        # learning rate, momentum influence and regularization influence
        self.momentumRate = momentumRate
        self.regularizationRate = regularization_rate


        # You can have better initialization here
        uniform_sample_margin = np.divide(np.sqrt(6), np.sqrt(nOut + nIn))
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            # glorot & bengio 2010 suggest setting up the weights depending on
            # number of neurons in the current layer and in the one below.
            self.weights = rns.uniform(size=(nOut, nIn + 1)) \
                           * uniform_sample_margin - 2 * uniform_sample_margin
            # set bias to 0, initially
            self.weights[0] = 0
        else:
            self.weights = weights

        self.isClassifierLayer = isClassifierLayer

        # Some handy properties of the layers
        self.size = self.nOut
        self.shape = self.weights.shape

    def forward(self, layerInput):
        """
        Compute forward step over the input using its weights

        Parameters
        ----------
        layerInput : ndarray
            a numpy array (1,nIn + 1) containing the input of the layer

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        self.output = []
        self.input = np.append([1], np.array(layerInput))
        preactivation = []
        for i in range(0, self.nOut):
            try:
                preactivation.append(np.dot(self.input, self.weights[i]))
            except ValueError as e:
                print("Failed to compute the activation of the following dot product: "
                      + str(self.input) + " and " + str(self.weights[i]))
        if self.isClassifierLayer:
            self.output = self.activation(preactivation)
        else:
            self.output = map(lambda a: self.activation(a), preactivation)
        return self.output

    # here's where the magic happens ^^
    def computeDerivative(self, nextDerivatives, nextWeights):
        """
        Compute the derivatives (backward pass) as follows:
        1.) First we calculate the propagated output error:
        dE_dy_i = expected_i - net_i
                = sum_{ j in {1 .. nIn}} dE_dx_j * w_ij}
                = sum_{ j in {1 .. nIn}} nextDerivatives[j] * nextWeights[j][i], i in {1 .. nOut}
        2.) from this we can calculate the new derivatives out of the
        activation function which we return to the caller together with the current
        weights:
        dE_dx_i = dE_dy_i * dy_i_dx_i
                = dE_dy_i * y_i * (1 - y_i)
                = dE_dy_i * self.activationPrime(self.output[i]), i in {1 .. nOut}
        3.) and update the weights of this layer:
        self.delta_i = dE_dx_i * dx_i_dw_i
                     = dE_dx_i * self.input, i in {1 .. nOut}
        self.updateWeights(self.learningRate)

        If isClassifierLayer is set to true, then expect precomputed dE_dy in the
        nextDerivatives argument and skip 1.).

        Parameters
        ----------
        nextDerivatives: ndarray
            a numpy array containing the derivatives from next layer
            (aka dE_dx_i for i in {1 .. nOut})
        nextWeights : ndarray
            a numpy array containing the weights from next layer

        Returns
        -------
        ndarray :
            a numpy array containing the partial derivatives on this layer
        """
        newDerivatives = []
        oldWeights = []
        for i in range(0, self.nOut):
            # 1.) First we calculate the propagated output error:
            if self.isClassifierLayer:
                # see the _tune_net_parameters description in ffnn.py
                dE_dx_i = nextDerivatives[i]
            else:
                dE_dy_i = 0
                for j in range(0, len(nextDerivatives)):
                    # when looking for contributions  from the layer above, skip the bias! ...[i + 1]
                    dE_dy_i += nextDerivatives[j] * nextWeights[j][i + 1]
                # 2.) from this we can calculate the new derivatives
                dE_dx_i = dE_dy_i * self.activationPrime(self.output[i])
            newDerivatives.append(dE_dx_i)
            oldWeights.append(self.weights[i])
            # 3.) add derivatives with respect to weights
            temp = dE_dx_i * self.input
            self.gradient_matrix[i] += temp
        return np.array(newDerivatives), np.array(oldWeights)

    def updateWeights(self, learningRate):
        """
        Update the weights of the layer.

        We call this method from outside the layer, so that we can decide if we do sgd, or
        batch/minibatch update.
        """
        for i in range(0, self.nOut):
            # L2 regularization
            regularizetion_contribution = (self.regularizationRate) * 2 * self.weights[i]
            weight_gradient_contribution = (-learningRate) * (self.gradient_matrix[i] + regularizetion_contribution)
            self.gradient_matrix[i] = np.zeros((self.nIn + 1,)) # so that the gradient does not continue to be updated indefinitely!!
            # momentum
            momentum_contribution = self.momentumRate * self.momentum_matrix[i]
            self.momentum_matrix[i] = weight_gradient_contribution
            if np.isnan(weight_gradient_contribution).any():
                raise ValueError("Through the roof! The update value for the "
                                 "layer weights just exploded... " + str(self.gradient_matrix) +
                                 " does not bode well with a learning rate of " + str(learningRate))
            self.weights[i] += weight_gradient_contribution + momentum_contribution

