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
        accuracy = []
        error_progresion = []
        self._initialize_plot()
        legend_exists = False

        #Train for some epochs if the error is not 0
        while not learned:
            hypothesis = np.array(list(map(self.classify,
                                           self.trainingSet.input)))
            d = np.array(list(map(self.fire,
                              self.trainingSet.input)))
            totalError = loss.calculateError(np.array(self.trainingSet.label),
                                        hypothesis)
            n_X = len(hypothesis)
            #print("Error now is: %f", totalError)
            if totalError != 0:
                # E(w) = 1/|X| * sigma_{x in X} (y(wx) - d)^2
                # where y(wx) = sigmoid(wx)
                # dE/dy = 1/|X| * sigma_{x in X} 2(y(wx) - d)
                dE_dy = (2.0 / n_X) * \
                        (np.array(self.trainingSet.label) - d)
                # now we need:
                # dE/dx = 1/|X| * sigma_{x in X} 2(y(wx) - d) * y'
                # wobei y'(wx) = y(wx) * (1-y(wx)) =: sigmoid_prime(wx)
                sigmoid_gradient_contributions = map(Activation.sigmoidPrime,
                                                     d)
                dE_dx = [a * b for a, b in
                         zip(dE_dy, sigmoid_gradient_contributions)]
                self.updateWeights(dE_dx)

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

    def updateWeights(self, dE_dx):
        weight_gradient_contributions = np.array([a * b for a, b in
                                                 zip(dE_dx, self.trainingSet.input)])
        total_gradient = np.sum(weight_gradient_contributions)
        self.weight += self.learningRate * total_gradient

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        return Activation.sigmoid(np.dot(np.array(input), self.weight))

    def _initialize_plot(self):
        pl.ion()

    def _update_plot(self, iteration, accuracy, error_progresion, legend_exists):
        x = range(iteration)
        pl.xlabel(u"Epochs")
        pl.xlim(0, self.epochs)
        pl.ylim(0, 1.0)
        pl.plot(x, accuracy, 'g-', label='accuracy')
        pl.plot(x, error_progresion, 'r-', label='mean sqared \nerror')
        if not legend_exists:
            # Now add the legend with some customizations.
            legend = pl.legend(loc='upper right')

            # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
            # frame = legend.get_frame()
            # frame.set_facecolor('0.90')

            # Set the fontsize
            for label in legend.get_texts():
                label.set_fontsize('large')

            for label in legend.get_lines():
                label.set_linewidth(1.5)  # the legend line width
            pl.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        pl.show()
        pl.pause(0.01)
        return True