
import time

import numpy as np

from util.activation_functions import Activation

#from model.layer import Layer


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
                 activation='softmax', isClassifierLayer=True):

        # Get activation function from string
        # Notice the functional programming paradigms of Python + Numpy
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)
        self.activationPrime = Activation.getDerivative(self.activationString)

        self.nIn = nIn
        self.nOut = nOut

        # Adding bias
        self.input = np.ndarray((nIn+1, 1))
        self.input[0] = 1
        self.output = np.ndarray((nOut, 1))
        self.delta = np.zeros((nOut, 1))

        # You can have better initialization here
        # wij means the weight from Input(j) to the Output(i)
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            self.weights = rns.uniform(size=(nOut, nIn + 1))-0.5
        else:
            self.weights = weights

        self.isClassifierLayer = isClassifierLayer

        # Some handy properties of the layers
        self.size = self.nOut
        self.shape = self.weights.shape

    def forward(self, inputN):
        """
        Compute forward step over the input using its weights

        Parameters
        ----------
        input : ndarray
            a numpy array (1,nIn + 1) containing the input of the layer

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        pass
        self.input[1:self.nIn+1,0] = inputN
        for i in range(0, self.nOut):
            self.output[i,0] = self.activation(np.dot(self.weights[i],
                                                      self.input))
        return self.output

    def computeDerivative(self, nextDerivatives, nextWeights):
        """
        Compute the derivatives (back)

        Parameters
        ----------
        nextDerivatives: ndarray
            a numpy array containing the derivatives from next layer
        nextWeights : ndarray
            a numpy array containing the weights from next layer
        Returns
        -------
        ndarray :
            a numpy array containing the partial derivatives on this layer
        """
        for i in range(0, self.nOut):
            dE_dy_i = np.dot(nextDerivatives,np.array(nextWeights)[:, i+1])
            dE_dx_i = dE_dy_i * self.activationPrime(self.output[i])
            self.delta[i] = dE_dx_i
        return self.delta

    def updateWeights(self, val):
        """
        Update the weights of the layer
        """
        #update weights actually always outside the layer class??
        for i in range(0, self.nOut):
            self.weights[i] += val[i]
