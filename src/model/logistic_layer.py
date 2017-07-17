
import time

import numpy as np

from util.activation_functions import Activation
# from model.layer import Layer


class LogisticLayer:
    """
    A layer of perceptrons acting as the output layer

    Parameters
    ----------
    nIn: int: number of units from the previous layer (or input data)
    nOut: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    isClassifierLayer: bool:  to do classification or regression

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
    delta : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, nIn, nOut, weights=None,
                 activation='softmax', isClassifierLayer=True, learningRate=0.01):

        # Get activation function from string
        # Notice the functional programming paradigms of Python + Numpy
        self.learningRate = learningRate
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)
        self.activationPrime = Activation.getDerivative(self.activationString)

        self.nIn = nIn
        self.nOut = nOut

        # Adding bias
        self.input = np.ndarray((nIn+1, 1))
        self.input[0] = 1
        self.output = np.ndarray((nOut, 1))
        self.delta = []

        # You can have better initialization here
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            self.weights = rns.uniform(size=(nOut, nIn + 1))-0.5
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
        for i in range(0, self.nOut):
            self.output.append(self.activation(np.dot(self.input,
                                                      self.weights[i])))
        return self.output

    # here's where the magic happens ^^
    def computeDerivative(self, nextDerivatives, nextWeights):
        """
        Compute the derivatives (backward pass) as follows:
        1.) First we calculate the propagated output error:
        dE_dy_i = expected_i - net_i
                = sum_{ j in {1 .. nIn}} dE_dx_j * w_ij}
                = sum_{ j in {1 .. nIn}} nextDerivatives[j] * nextWeights[i][j]
                = nextDerivatives * nextWeights[i], i in {1 .. nOut}
        2.) from this we can calculate the new derivatives out of the
        activation function which we return to the caller together with the current
        weightss:
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
                dE_dy_i = nextDerivatives[i]
            else:
                dE_dy_i = nextDerivatives * nextWeights[i]
            # 2.) from this we can calculate the new derivatives
            dE_dx_i = dE_dy_i * self.activationPrime(self.output[i])
            newDerivatives.append(dE_dx_i)
            oldWeights.append(self.weights[i])
            # 3.) add derivatives with respect to weights
            self.delta.append(np.dot(dE_dx_i, self.input))
        # update weights
        self.updateWeights()
        return np.array(newDerivatives), np.array(oldWeights)

    def updateWeights(self):
        """
        Update the weights of the layer
        """
        # always update weights from outside the layer class
        for i in range(0, self.nOut):
            self.weights[i] += self.learningRate * np.array(self.delta[i])
