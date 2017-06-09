# -*- coding: utf-8 -*-

import matplotlib.pylab as pl
import sys
import logging

import numpy as np
import util.loss_functions as erf

from util.activation_functions import Activation
from model.classifier import Classifier
from model.logistic_layer import LogisticLayer
from sklearn.metrics import accuracy_score

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50,
                 activation = 'sigmoid', error = 'mse'):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(1, self.trainingSet.input.shape[1])
        weight_Plus_bias = np.insert(self.weight, 0, 1, axis=1)
        self.weight = weight_Plus_bias

        # Choose the error function
        self.errorString = error
        self._initialize_error(error)

        #initialize also the layer
        self.layer = LogisticLayer(nIn = self.trainingSet.input.shape[1],
                                   nOut = 1, activation = 'sigmoid',
                                   weights = weight_Plus_bias)

    def _initialize_error(self, error):
        if error == 'absolute':
            self.erf = erf.AbsoluteError()
        elif error == 'mse':
            self.erf = erf.MeanSquaredError()
        elif error == 'sse':
            self.erf = erf.SumSquaredError()
        else:
            raise ValueError('Cannot instantiate the requested error function:' + error + 'not available')


    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        #from util.loss_functions import MeanSquaredError
        #loss = MeanSquaredError()
        learned = False
        iteration = 0
        accuracy = []
        pl.ion()
        input_Plus_bias = np.insert(self.trainingSet.input, 0, 1, axis=1)

        #Train for some epochs if the error is not 0
        while not learned:
            totalError = 0
            derivatives = []
            hypothesis = np.array(list(map(self.classify,
                                           self.trainingSet.input)))
            totalError = self.erf.calculateError(np.array(self.trainingSet.label),
                                        hypothesis)
            #print("Error now is: %f", totalError)
            if totalError != 0:
                output = np.array([])
                for i in range(0, self.trainingSet.input.shape[0]):
                    if i == 0:
                        output = self.layer.forward(self.trainingSet.input[i])
                    else:
                        np.append(output,
                              self.layer.forward(self.trainingSet.input[i]),
                              axis = 0)
                dE_dy =self.erf.calculatePrime(np.array(self.trainingSet.label),
                                               output)
                # for only one neuron set the weight as [0,1]
                dE_dx = self.layer.computeDerivative([dE_dy], [[0, 1]])
                dE_dw = dE_dx * input_Plus_bias
                self.layer.updateWeights(dE_dw)

            iteration += 1

            if verbose:
                logging.info("Epoch: %i; Error: %f", iteration, totalError)
            if totalError == 0 or iteration >= self.epochs:
                learned = True
           # accuracy.append(accuracy_score(self.trainingSet.label, hypothesis))
            x = range(iteration)
           # pl.xlabel(u"Epochs")
           # pl.ylabel(u"Accuracy")
           # pl.xlim(0, self.epochs)
           # pl.ylim(0, 1.0)
           # pl.plot(x, accuracy, 'k')
           # pl.show()
           # pl.pause(0.01)


    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        return self.layer.forward(testInstance) >= 0.5

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self, grad):
        self.weight += self.learningRate*grad

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))
