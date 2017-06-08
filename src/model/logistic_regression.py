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
        weight = np.random.rand(self.trainingSet.input.shape[1] + 1) * 6 - 3

        self.erString = error
        self._initialize_error(error)

        # if we want more than just one neuron per layer, add the separate
        # neuron weights as a separate vector in the weights parameter
        self.layer = LogisticLayer(nIn=self.trainingSet.input.shape[1],
                                   nOut=1,
                                   activation=activation,
                                   weights=np.array([weight]))


    def _initialize_error(self, error):
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
        learned = False
        iteration = 0
        accuracy = []
        error_progresion = []
        self._initialize_plot()
        legend_exists = False

        old_score = 0  # for early stopping
        while not learned:
            derivatives = []    # contains the gradients for the training set
            d = np.array(self.trainingSet.label) # the desired output for the ts

            # whenever we do a forward pass we also have to do a backw pass.
            # we compute the error, do the update and cross_validate after the loop
            for i in range(0, self.trainingSet.input.shape[0]):
                dE_dw = self._get_gradient(d[i], self.trainingSet.input[i])
                derivatives.append(dE_dw)

            # compute error & update
            output = list(map(self.classify, self.trainingSet.input))
            totalError = self.erf.calculateError(d, output)
            if totalError != 0:  # update weights if necessary
                self.updateWeights(derivatives)
            error_progresion.append(totalError)

            # validation
            validation_output = np.array(list(map(self.classify,
                                                  self.validationSet.input)))
            new_score = accuracy_score(self.validationSet.label, validation_output)
            accuracy.append(accuracy_score(self.validationSet.label,
                                           validation_output))
            
            # stop conditions
            iteration += 1
            if totalError == 0 or iteration >= self.epochs or old_score > new_score:
                learned = True
            old_score = new_score

            #logging
            if verbose:
                logging.info("Epoch: %i; Error: %f", iteration, totalError)
            # plots
            legend_exists= self._update_plot(iteration,
                              accuracy, error_progresion,
                              legend_exists)




    def _get_gradient(self, d, input):
        y = self.layer.forward(input)
        dE_dy = self.erf.calculateErrorPrime(d, np.array(y[0]))
        # if toplayer, there are no weights after activation function
        # -> weights are all one
        dE_dx = self.layer.computeDerivative([dE_dy], np.ones(self.layer.nOut))

        dE_dw = dE_dx * input
        return dE_dw



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
        # there is just 1 output, for now...
        return self.fire(testInstance)[0] >= 0.5

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
        total_gradient = []
        for i in range(0, self.layer.nOut):
            total_gradient.append(self.learningRate * np.sum(dE_dw))
        self.layer.updateWeights(total_gradient)

    def fire(self, input):
        # Look at how we change the activation function here!!!!
        # Not Activation.sign as in the perceptron, but sigmoid
        #return self.activation(np.dot(np.array(input), self.weight))
        return self.layer.forward(input)

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