# -*- coding: utf-8 -*-

import copy as cp
import logging
import sys

import numpy as np
from sklearn.metrics import accuracy_score

import util.loss_functions as erf
from model.classifier import Classifier
from model.ffnn_layer import FFNNLayer
from report.plotter import Plotter

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class FFNN(Classifier):
    """
    Class constructor. Initializes a fully connected feed forward neural
    network with a softmax output layer.

    The number of hidden units and input units can be specified through the
    network_representation parameter, which should be a list of integers. Each
    entry in the list triggers the instantiation of a layer with a number of
    nodes equal to the list entry. The leftmost entry specifies the number of
    input nodes. Every subsequent connection specifies the number of nodes in
    the hidden layers above.

    Each non-output node in the network has a sigmoid activation function.

    Learning and generalization methods implemented: momentum, annealing of the
    learning rate (high in the beginning for exploration, lower further on for
    convergence), L2 regularization.

    The update policy is either batch update or sgd. The batch size can be
    controlled through the appropriate parameter. batch_size=1 (sgd) by default.

    The network keeps track of the weights configuration that produced the best
    output, and uses this configuration for evaluation

    (Some) Attributes
    ----------
    learning_rate : the factor of the gradient update
    momentum : influence of the previous iteration's gradient on the current update
    regularization : factor of the L2 regularization term
    erf : the error function; currently, just crossentropy is supported
    network_representation : list of integers descibing the number of units in non-
        output layers
    batch_size : specifies the size of the mini-batch for batch update (1 -> sgd)
    best_model : contains the layer configuration yielding the best validation result thus far
    """

    def __init__(self, train,
                 learningRate=0.01, momentum=0.005, regularization_rate=0.5, epochs=50, error='mse',
                 batch_size=1, network_representation=None, verbose=True, normalized=False):
        """
        Class constructor. Initializes the fully connected feed forward neural network
        """
        # hyperparameters
        self.learningRate = learningRate
        self.epochs = epochs
        self.momentum = momentum
        self.regularization = regularization_rate
        # update policy
        self.batch_size = batch_size
        # architecture parameters
        self.erString = error
        self.erf = self._initialize_error(error)
        self.layers = self._initialize_network(network_representation, momentum, regularization_rate,
                                               train.input.shape[1], train.label.shape[1])
        # for remembering the layer configuration yielding the best results
        self.best_validation_set_accuracy = 0
        self.best_model = None
        self.learned = False
        # control parameters: mainly for following the learning procedure (plotting, logging)
        self.current_training_error = -1
        self.current_validation_error = -1
        self.current_training_accuracy = -1
        self.current_validation_accuracy = -1
        self.verbose = verbose
        # data normalization parameters
        self.normalized = normalized
        if normalized:
            self.ts_mean, self.ts_variance = self._get_normalization_statistics(train.input)
        # string describing the current configuration of hyper-parameters
        self.model_descriptor = self._set_descriptor_string(network_representation)

    @staticmethod
    def _get_normalization_statistics(training_set_input):
        """
        Computes the mean and variance of the training set for normalization.

        'Reasoning: A model shall be applied on unseen data which is in general not
        available at the time the model is built. The validation process (including
        data splitting) simulates this. So in order to get a good estimate of the
        model quality (and generalization power) one needs to restrict the calculation
        of the normalization parameters (mean and variance) to the training set.',
        source:
        https://stats.stackexchange.com/questions/77350/perform-feature-normalization-before-or-within-model-validation

        :param training_set_input: The training set data.
        :return: The mean and variance.
        """
        mean = np.array(training_set_input).mean(axis=0)
        v = np.array(training_set_input).var(axis=0)
        # some input dimensions have 0 mean and 0 variance;
        # to prevent 0 divide error in normalization, we replace 0 in variance with the machine epsilon
        variance = np.array(list(map(lambda x: x if x != 0 else np.finfo(float).eps, v)))
        return mean, variance

    def _set_descriptor_string(self, network_representation):
        """
        Initializes the string describing this model's hyperparameters.

        :param network_representation: The list representing the network.
        :return: The descriptor string.
        """
        norm = "_0m1vNormalized" if self.normalized else ""
        return "ffnn_" + str(network_representation) + "nodes_" \
               + str(self.epochs) + "epochs_" \
               + str(self.learningRate) + "lr_" \
               + str(self.momentum) + "m_" \
               + str(self.batch_size) + "batchSize" + str(self.regularization) \
               + "L2Regularized" + norm

    @staticmethod
    def _initialize_network(network_representation, momentum, regularization_rate, nInInput, nOutOutput):
        """
        Initializes the network in accordance with the parameters specified in
        network_representation.

        This is a list of integers representing the number of nodes per layer
        (aka logistic_layer.nOut). The first entry in the layer specifies the
        number of nodes directly connected with the input. The last entry in
        the list specifies the number of output nodes. (layers get constructed
        bottom up)

        The network is a fully connected feed forward network where the outputs
        of a layer make up the input of each node of the next layer. Output nodes
        use softmax. Inner/input nodes use sigmoid.

        :param network_representation: The list of nodes per layer.
        :param momentum: Scalar of the previous' iteration gradient contribution.
        :param regularization_rate: The L2 regularization rate.
        :param nInInput: The number of dimensions of the input vectors.
        :param nOutOutput: The number of output units.
        :return: the list of layers.
        """
        layers = []
        nIn = nInInput
        nLayers = len(network_representation) + 1
        for i in range(0, nLayers):
            if i != nLayers - 1:                    # inner/input layer
                layer = FFNNLayer(nIn=nIn,
                                  nOut=network_representation[i],
                                  activation="sigmoid",
                                  weights=None,
                                  isClassifierLayer=False,
                                  regularization_rate=regularization_rate,
                                  momentumRate=momentum)
                nIn = network_representation[i]
            else:                                   # output layer
                nOut = nOutOutput
                layer = FFNNLayer(nIn=nIn,
                                  nOut=nOut,
                                  activation="softmax",
                                  weights=None,
                                  regularization_rate=regularization_rate,
                                  momentumRate=momentum)
            layers.append(layer)
        return layers

    def train(self, trainingSet, validationSet):
        """
        Trains the Feed Forward Neural Network.

        :param trainingSet: The labeled training set.
        :param validationSet: The labeled validation set.
        """
        epoch = 0
        pl = Plotter(self.epochs, self.model_descriptor, self.erString)
        input_training = trainingSet.input
        while not self.learned:
            n_inputs = input_training.shape[0]
            n_inBatch = 0
            for i in range(0, n_inputs):
                y = self.fire(input_training[i])
                if not np.array_equal(y, np.array(trainingSet.label)[i]):  # skip correctly classified inputs
                    n_inBatch += 1
                    self._update_net_gradient(np.array(trainingSet.label)[i], y)
                    if (n_inBatch % self.batch_size == 0) or (i == n_inputs - 1):
                        self._update_net_weights()
            epoch += 1
            # update the control variables and output information
            self._update_and_display_control_parameters(pl, epoch, trainingSet, validationSet)
            if self.best_validation_set_accuracy < self.current_validation_accuracy:
                self.best_validation_set_accuracy = self.current_validation_accuracy
                self.best_model = cp.deepcopy(self.layers)
            # stop condition
            if self.current_training_error == 0 or epoch >= self.epochs:
                self.learned = True

    def _update_net_weights(self):
        """
        Calls the layer weight update methods to update their weights from the stored
        gradients, according to specified hyperparameters. An annealed learning rate is used.
        """
        for layer in reversed(self.layers):
            layer.updateWeights(self.learningRate)

    def _normalize(self, network_input):
        interim = (network_input - self.ts_mean) / self.ts_variance
        return np.array(list(map(lambda x: 0 if np.isnan(x) else x, interim)))

    def _update_and_display_control_parameters(self, pl, epoch, trainingSet, validationSet):
        """
        Updates the instance variables keeping track of the model's performance and outputs
        this data through plots and logging.

        :param pl: The plotter.
        :param epoch: The current iteration.
        :param trainingSet: The training set.
        :param validationSet: The validation set.
        """
        # probability outputs
        o_t = list(map(lambda x: self.fire(x), trainingSet.input))
        o_v = list(map(lambda x: self.fire(x), validationSet.input))
        # errors
        self.current_training_error = self.erf.calculateError(trainingSet.label, o_t)
        self.current_validation_error = self.erf.calculateError(validationSet.label, o_v)
        # class outputs
        classes_t = np.array(list(map(self._to_descrete, o_t)))
        classes_v = np.array(list(map(self._to_descrete, o_v)))
        # accuracy
        self.current_training_accuracy = accuracy_score(trainingSet.label, classes_t)
        self.current_validation_accuracy = accuracy_score(validationSet.label, classes_v)
        # plotting
        pl.update_control_information(self.current_training_accuracy,
                                      self.current_validation_accuracy,
                                      self.current_training_error,
                                      self.current_validation_error)
        pl.update_plot()
        # logging
        if self.verbose:
            logging.info("Epoch: %i; Test set error: %f", epoch, self.current_training_error)

    def _update_net_gradient(self, t, y):
        """ TODO: Comment"""
        if self.erString == 'crossentropy' and self.layers[-1].activationString == 'softmax':
            dE_dx = self.erf.calculateErrorPrime(t, np.array(y))
            newDerivatives, oldWeights = (dE_dx, None)
        elif self.layers[-1].activationString == 'sigmoid':
            dE_dy = self.erf.calculateErrorPrime(t, np.array(y[0]))
            newDerivatives, oldWeights = (dE_dy, None)
        else:
            raise ValueError("Illegal activation&error-function combination!")
        for layer in reversed(self.layers):
            newDerivatives, oldWeights = layer.computeDerivative(
                newDerivatives, oldWeights)

    def classify(self, testInstance):
        """
        Classify a single instance.

        :param testInstance: The instance to predict the label for.
        :return: The vector indicating the most probable label.
        """
        return self._to_descrete(self.fire(testInstance))

    @staticmethod
    def _to_descrete(class_descriptor):
        """
        Transforms a vector of probabilities into a vector containing 1 for
        the entry with the highest probability and 0 otherwise.

        :param class_descriptor: The vector of probabilities.
        :return: The vector indicating the most probable label.
        """
        index = np.argmax(class_descriptor)
        for i in range(0, len(class_descriptor)):
            if i == index:
                class_descriptor[i] = 1.
            else:
                class_descriptor[i] = 0.
        return class_descriptor

    def evaluate(self, test=None):
        """
        Evaluate a whole dataset.

        :param test: The network input.
        :return: The list of predicted classes (multiclass-labels).
        """
        if self.learned:
            self.layers = self.best_model
        return np.array(map(self.classify, test))

    def fire(self, network_input):
        """
        Triggers the forward pass of the feed forward network, layer by layer.
        Note that the network input has to match the "nIn" instance variable of the
        first layer.

        :param network_input: The network input.
        :return: The network output.
        """
        layer_input = self._normalize(network_input) if self.normalized else network_input
        layer_output = None
        for layer in self.layers:
            layer_output = layer.forward(layer_input)
            layer_input = cp.deepcopy(layer_output)
        return layer_output

    @staticmethod
    def _initialize_error(error):
        """
        Assigns the error function to the erf field, dependent on the error string.
        Raises a value error if the error string is unknown
        (only errors defined in the util.loss_functions module are permitted).

        :param error: The error string.
        :return: The error function (sic!).
        """
        if error == 'absolute':
            return erf.AbsoluteError()
        elif error == 'different':
            return erf.DifferentError()
        elif error == 'mse':
            return erf.MeanSquaredError()
        elif error == 'sse':
            return erf.SumSquaredError()
        elif error == 'bce':
            return erf.BinaryCrossEntropyError()
        elif error == 'crossentropy':
            return erf.CrossEntropyError()
        else:
            raise ValueError('Cannot instantiate the requested '
                             'error function: ' + error + 'not available')
