# -*- coding: utf-8 -*-

import matplotlib.pylab as pl
import sys
import logging

import numpy as np
import copy as cp
import util.loss_functions as erf

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
    learningRate : positive float;
    epochs : positive int;

    trainingSet :
    validationSet :
    testSet :

    activation_string : string; the activation function specifier
    erString : string; the error function specifier
    erf : loss_function; the error function
    layers : list of Logistic_layer objects; the network layer from
        bottom (index 0 in the list) to top(/output; last index in the list.)
    """

    def __init__(self, train, valid, test,
                 learningRate=0.01, epochs=50,
                 activation_string='sigmoid',
                 error='mse',
                 network_representation=None):
        """
        Class constructors. Initializes a fully connected feed forward neural
        network with the following a layer architecture design implicitly defined
        by the network_representation parameter.

        This class manages the between layer (intra-network) dataflow (forward as
        well as backward), and acts as a communication interface for tuning the
        network and specifying the network data (training set, validation set,
        test set).

        Each node in the network will share the same activation function.

        Each layer's forward and backward pass is managed internally in the
        logistic_layer class, which models a generic layer.

        :param train: The training set.
        :param valid: The validation set.
        :param test: The test set.
        :param learningRate: The rate of descent.
        :param epochs: The number of iterations.
        :param activation_string: The activation function identifier.
        :param error: The error function identifier.
        :param network_representation: The list of node numbers per layer.
        """

        # tuning globals
        self.learningRate = learningRate
        self.epochs = epochs
        # data globals
        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        # design globals
        self.activation_string = activation_string
        self.erString = error
        # architecture initialization
        self._initialize_error(error)
        self._initialize_network(network_representation)

    def _initialize_network(self, network_representation):
        """
        Initializes the network in accordance with the parameters specified in
        network_representation.

        This is a list of integers representing the number of nodes per layer
        (aka logistic_layer.nOut). The first entry in the layer specifies the
        number of nodes directly connected with the input. The last entry in
        the list specifies the number of output nodes. (layers get constructed
        bottom up)

        The network is a fully connected feed forward network where the outputs
        of a layer make up the input of each node of the next layer and all
        nodes use the same activation specified in this class's constructor.

        :param network_representation: The list of nodes per layer.
        :return: void.
        """
        self.layers = []
        nIn = self.trainingSet.input.shape[1]
        for i in range(0, len(network_representation)):
            # initialize layer weights:
            # a separate array of weights for each node in the layer.
            weights = []
            for j in range(0, network_representation[i]):
                weight = np.random.rand(nIn + 1) - 0.7    # tweak initial weights here!
                weights.append(weight)
            layer = LogisticLayer(nIn=nIn,
                                  nOut=network_representation[i],
                                  activation=self.activation_string,
                                  weights=np.array(weights),
                                  learningRate=self.learningRate)
            self.layers.append(layer)
            nIn = network_representation[i]

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
            d = np.array(self.trainingSet.label)    # the desired output for the ts

            # whenever we do a forward pass we also have to do a backw pass.
            # we compute the error, do the update and cross_validate after the loop
            for i in range(0, self.trainingSet.input.shape[0]):
                self._get_gradient(d[i], self.trainingSet.input[i])

            # compute error
            output = list(map(lambda x: self.fire(x)[0], self.trainingSet.input))
            totalError = self.erf.calculateError(d, output)
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
        y = self.fire(input)
        dE_dy = self.erf.calculateErrorPrime(d, np.array(y[0]))
        newDerivatives, oldWeights = ([dE_dy], None)
        for layer in reversed(self.layers):
            newDerivatives, oldWeights = layer.computeDerivative(
                newDerivatives, oldWeights)

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

    def fire(self, network_input):
        """
        Triggers the forward pass of the feed forward network, layer by layer.
        Note that the network input has to match the "nIn" instance variable of the
        first layer.

        :param network_input: The network input.
        :return: The network output.
        """
        layer_input = network_input
        for layer in self.layers:
            layer_output = layer.forward(layer_input)
            layer_input = cp.deepcopy(layer_output)
        return layer_output

    # ===================== PRIVATE HELPERS =========================================
    @staticmethod
    def _initialize_plot():
        """
        Initializes the runtime plotting functionality by setting the interactive
        mode to "on".
        """
        pl.ion()

    def _update_plot(self, iteration, accuracy, error, legend_exists):
        """
        Updates the runtime plot with the new accuracy and error values.
        If the legend has not yet been added to the plot, it will also be
        initialized.

        :param iteration: Defines the new x-axis range.
        :param accuracy: The new accuracy value.
        :param error: The new error value.
        :param legend_exists: True if the legend has already been added,
        False otherwise.

        :return: Returns True to indicate the legend has been added.
        """
        x = range(iteration)
        pl.xlabel(u"Epochs")
        pl.figure(1)
        sp1 = pl.subplot(211)
        pl.xlim(0, self.epochs)
        pl.ylim(0, 1.0)
        pl.plot(x, accuracy, 'g-', label='accuracy')

        sp2 = pl.subplot(212)
        pl.xlim(0, self.epochs)
        pl.ylim(0, np.max(error))
        pl.plot(x, error, 'r-', label=(self.erString + ' error'))
        if not legend_exists:
            # Now add the legend with some customizations.
            sp1.legend(loc='upper right')
            sp2.legend(loc='upper right')
        pl.show()
        pl.pause(0.01)
        return True

    def _initialize_error(self, error):
        """
        Assigns the error function to the erf field, dependent on the error string.
        Raises a value error if the error string is unknown
        (only errors defined in the util.loss_functions module are permitted).

        :param error: The error string.
        """
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
