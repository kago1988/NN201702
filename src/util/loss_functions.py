# -*- coding: utf-8 -*-


"""
Loss functions.
"""

import numpy as np

from abc import ABCMeta, abstractmethod, abstractproperty


class Error:
    """
    Abstract class of an Error
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def errorString(self):
        pass

    @abstractmethod
    def calculateError(self, target, output):
        # calculate the error between target and output
        pass

    @abstractproperty
    def calculateErrorPrime(self, target, output):
        # error function derivative with respect to neuron output
        pass


class AbsoluteError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """
    def errorString(self):
        self.errorString = 'absolute'

    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        return np.sum(np.abs(target - output))

    def calculateErrorPrime(self, target, output):
        # error function derivative with respect to neuron output
        return -output


class DifferentError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """
    def errorString(self):
        self.errorString = 'different'

    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        return target - output

    def calculateErrorPrime(self, target, output):
        # error function derivative with respect to neuron output
        return -output

class MeanSquaredError(Error):
    """
    The Loss calculated by the mean of the total squares of differences between
    target and output.
    """
    def errorString(self):
        self.errorString = 'mse'

    def calculateError(self, target, output):
        # MSE = 1/n*sum (i=1 to n) of (target_i - output_i)^2)
        # muss zuerst target und output zu Array von Numpy wechseln!!
        # hier die Type von target und output sind Array von Numpy!!
        output_errors = np.average((target - output)**2)
        return output_errors

    def calculateErrorPrime(self, target, output):
        return np.array(target) - output


class SumSquaredError(Error):
    """
    The Loss calculated by the sum of the total squares of differences between
    target and output.
    """
    def errorString(self):
        self.errorString = 'sse'

    def calculateError(self, target, output):
        # SSE = 1/2*sum (i=1 to n) of (target_i - output_i)^2)
        # die Type von target und output sind Array aus Numpy!!!
        squares = (output - target)**2
        return 0.5 *  np.sum(squares)

    def calculateErrorPrime(self, target, output):
        return (2.0 / len(output)) * (np.array(target) - output)

class BinaryCrossEntropyError(Error):
    """
    The Loss calculated by the Cross Entropy between binary target and
    probabilistic output (BCE)
    """
    def errorString(self):
        self.errorString = 'bce'

    def calculateError(self, target, output):
        pass

    def calculateErrorPrime(self, target, output):
        # error function derivative with respect to neuron output
        pass

class CrossEntropyError(Error):
    """
    The Loss calculated by the more general Cross Entropy between two
    probabilistic distributions.
    """
    def errorString(self):
        self.errorString = 'crossentropy'

    def calculateError(self, target, output):
        pass

    def calculateErrorPrime(self, target, output):
        # error function derivative with respect to neuron output
        pass