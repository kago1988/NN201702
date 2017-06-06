# -*- coding: utf-8 -*-

import matplotlib.pylab as pl
import sys
import logging

import numpy as np

import util.loss_functions as erf

from util.activation_functions import Activation
from model.classifier import Classifier
from sklearn.metrics import accuracy_score

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """/home/kaze/Desktop/A_NN/NN_Praktikum_repos/NN201702
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

    def __init__(self, train, valid, test,
                 learningRate=0.01, epochs=50,
                 activation='sigmoid',
                 error='mse'):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values between -3 and 3
        # coresponding to the accelerated learning area for the sigmoid function
        self.weight = np.random.rand(self.trainingSet.input.shape[1]) * 6 - 3

        self.activation = Activation.getActivation(activation)
        self.activationPrime = Activation.getDerivative(activation)
        self.activationString = activation[0].upper() + activation[1:]

        self.erString = error

        if error == 'absolute':
            self.erf = erf.AbsoluteError()
        elif error == 'different':
            self.erf = erf.DifferentError()
        elif error == 'mse':
            self.erf = erf.MeanSquaredError()
        elif error == 'sse':
            self.erf = erf.SumSquaredError()
        elif error == 'bce':
            self.erf = erf.BinaryCrossEntropyError()
        elif error == 'crossentropy':
            self.erf = erf.CrossEntropyError()
        else:
            raise ValueError('Cannot instantiate the requested '
                             'error function: ' + error + 'not available')

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        from util.loss_functions import MeanSquaredError
        learned = False
        iteration = 0
        accuracy = []
        error_progresion = []
        self._initialize_plot()
        legend_exists = False

        #Train for some epochs if the error is not 0
        while not learned:
            hypothesis = np.array(list(map(self.classify,
                                           self.trainingSet.input)))
            net_output = np.array(list(map(self.fire,
                              self.trainingSet.input)))
            totalError = self.erf.calculateError(np.array(self.trainingSet.label),
                                                 hypothesis)

            #print("Error now is: %f", totalError)
            if totalError != 0:
                grad = self._get_gradient(net_output)
                self.updateWeights(grad)

            iteration += 1

            if verbose:
                logging.info("Epoch: %i; Error: %f", iteration, totalError)
            if totalError == 0 or iteration >= self.epochs:
                learned = True
            accuracy.append(accuracy_score(self.trainingSet.label, hypothesis))
            error_progresion.append(totalError)
            legend_exists= self._update_plot(iteration,
                              accuracy, error_progresion,
                              legend_exists)

    def _get_gradient(self, y):
        d = np.array(self.trainingSet.label)
        # E(w) = 1/|X| * sigma_{x in X} (y(wx) - d)^2
        # where y(wx) = sigmoid(wx)
        # dE/dy = 1/|X| * sigma_{x in X} 2(y(wx) - d)
        dE_dy = self.erf.calculateErrorPrime(d, np.array(y))
        # now we need:
        # dE/dx = 1/|X| * sigma_{x in X} 2(y(wx) - d) * y'
        # wobei y'(wx) = y(wx) * (1-y(wx)) =: sigmoid_prime(wx)
        sigmoid_gradient_contributions = map(self.activationPrime, y)
        dE_dx = [a * b for a, b in
                 zip(dE_dy, sigmoid_gradient_contributions)]
        weight_gradient_contributions = np.array([a * b for a, b in
                                                  zip(dE_dx, self.trainingSet.input)])
        return weight_gradient_contributions



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
        return self.fire(testInstance) >= 0.5

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

    def updateWeights(self, dE_dw):
        total_gradient = np.sum(dE_dw)
        self.weight += self.learningRate * total_gradient

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return self.activation(np.dot(np.array(input), self.weight))

    def _initialize_plot(self):
        pl.ion()

    def _update_plot(self, iteration, accuracy, error_progresion, legend_exists):
        x = range(iteration)
        pl.xlabel(u"Epochs")
        pl.figure(1)
        sp1 = pl.subplot(211)
        pl.xlim(0, self.epochs)
        pl.ylim(0, 1.0)
        pl.plot(x, accuracy, 'g-', label='accuracy')

        sp2 = pl.subplot(212)
        pl.xlim(0, self.epochs)
        pl.ylim(0, np.max(error_progresion))
        pl.plot(x, error_progresion, 'r-', label=(self.erString + ' error'))
        if not legend_exists:
            # Now add the legend with some customizations.
            legend1 = sp1.legend(loc='upper right')
            legend2 = sp2.legend(loc='upper right')
        pl.show()
        pl.pause(0.01)
        return True