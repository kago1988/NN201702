# -*- coding: utf-8 -*-
import matplotlib.pylab as pl
import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier
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

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small values
        self.weight = 0.01*np.random.randn(self.trainingSet.input.shape[1])

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        from util.loss_functions import MeanSquaredError
        loss = MeanSquaredError()

        learned = False
        iteration = 0
        target = np.array(self.trainingSet.label)
        input = np.array(self.trainingSet.input)
        accuracy = []
        pl.ion()

        while not learned:
            output = np.array(list(map(self.classify, input)))
            output_y = np.array(list(map(self.fire, input)))
            totalError = loss.calculateError(target, output_y)
            output_x = np.dot(input, self.weight)
            dE_dy = target - output_y
            #dE_dx = dE_dy*output_y*(1-output_y)
            dE_dx = dE_dy
            grad = np.dot(dE_dx, input)
            self.updateWeights(grad)

            iteration += 1

            if verbose:
                logging.info("Epoch: %i; Error: %f" , iteration, totalError)

            if totalError == 0 or iteration >= self.epochs:
                learned = True

            accuracy.append(accuracy_score(self.trainingSet.label, output))
            x = range(iteration)
            pl.xlabel(u"Epochs")
            pl.ylabel(u"Accuracy")
            pl.xlim(0, self.epochs)
            pl.ylim(0, 1.0)
            pl.plot(x, accuracy, 'k')
            pl.show()
            pl.pause(0.01)

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
        return self.fire(testInstance) > 0.5

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
        self.weight  += self.learningRate*grad
        return self.weight

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))
