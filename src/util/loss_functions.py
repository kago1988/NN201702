# -*- coding: utf-8 -*-


"""
Loss functions.
"""

import numpy as np
import math as m

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
        return target - output


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
        """
        Takes the single value output, and the single value desired output and
        computes the derivative contribution for the pairs.
        """
        return target - output

class BinaryCrossEntropyError(Error):
    """
    The Loss calculated by the Cross Entropy between binary target and
    probabilistic output (BCE)
    """
    def __init__(self):
        self.set_size = 1

    def errorString(self):
        self.errorString = 'bce'

    def calculateError(self, target, output):
        self.set_size = len(output)
        entropy = 0.0
        for i in range(0, len(output)):
            entropy += target[i] * np.log(output[i]) \
                       + (1.0 - target[i]) * np.log(1.0 - output[i])
        entropy = -(1.0 / self.set_size) * entropy
        return entropy

    def calculateErrorPrime(self, target, output):
        """
        Expects integer parameters!

        :param target: The desired output.
        :param output: The actual output.
        :return: The error derivative.
        """
        # error function derivative with respect to neuron output
        first_term = 0 if target == 0 else target * np.divide(1.0, output)
        if target == 1:
            second_term = 0
        else:
            # here we sometimes get zero if we overshoot terribly (as in we expect a 0
            # and we get a value infinitesimally close to one...)
            # print("t: " + str(target) + " <---> o: " + str(output))
            second_term = (1.0 - target) * np.divide(1.0, (1.0 - output)) * (-output)
            if np.isnan(second_term):
                error_string = 'Error is through the stratosfere and too close to one! ' \
                               'We expected a value close to 0 and got ' + str(output)
                raise ValueError(error_string)
        derivative = first_term + second_term
        #print((1.0 / self.set_size) * derivative)
        return - np.divide(1.0, self.set_size) * derivative

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