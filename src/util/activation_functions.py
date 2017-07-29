# -*- coding: utf-8 -*-

"""
Activation functions which can be used within neurons.
"""

import numpy as np

class Activation:
    """
    Containing various activation functions and their derivatives
    """

    @staticmethod
    def sign(netOutput, threshold=0):
        return netOutput >= threshold

    @staticmethod
    def sigmoid(netOutput):
        sig = np.divide(1.0, (1.0 + np.exp(-netOutput)))
        # if sig == 1: # this happens a lot!
        #    print("sigmoid spat a 1, which should not have happened, "
        #          "because we foolishly trusted the numpy exp and divide implementations... "
        #          "\nThe net output was " + str(netOutput) +
        #          "\nThe exp function result was " + str(np.exp(-netOutput)))
        return sig

    @staticmethod
    def sigmoidPrime(netOutput):
        # Here you have to code the derivative of sigmoid function
        # netOutput.*(1-netOutput)
        return netOutput*(1-netOutput)

    @staticmethod
    def tanh(netOutput):
        pass

    @staticmethod
    def tanhPrime(netOutput):
        # Here you have to code the derivative of tanh function
        pass

    @staticmethod
    def rectified(netOutput):
        return lambda x: max(0.0, x)

    @staticmethod
    def rectifiedPrime(netOutput):
        # Here you have to code the derivative of rectified linear function
        pass

    @staticmethod
    def identity(netOutput):
        return lambda x: x

    @staticmethod
    def identityPrime(netOutput):
        # Here you have to code the derivative of identity function
        pass

    @staticmethod
    def softmax(preactivation_values):
        """
        Receives the list of net outputs of a layer which it then uses to
        associate (class) probabilities to each one.

        :param preactivation_values: The vector of preactivation values.
        :return: The (class) probability distribution.
        """
        model_conditional_class_distribution = []
        exp_preactivation = map(lambda a: np.exp(a), preactivation_values)
        sum = 0
        for el in exp_preactivation:
            sum += el
        for i in range(0, len(exp_preactivation)):
            class_probability = np.divide(exp_preactivation[i], sum)
            if np.isnan(class_probability):
                # print("preactivation values: " + str(preactivation_values) +
                #       "\ncurrent softmax numerator: " + str(exp_preactivation[i]))
                model_conditional_class_distribution.append(0)
            else:
                model_conditional_class_distribution.append(class_probability)
        return model_conditional_class_distribution

    @staticmethod
    def getActivation(str):
        """
        Returns the activation function corresponding to the given string
        """

        if str == 'sigmoid':
            return Activation.sigmoid
        elif str == 'softmax':
            return Activation.softmax
        elif str == 'tanh':
            return Activation.tanh
        elif str == 'relu':
            return Activation.rectified
        elif str == 'linear':
            return Activation.identity
        else:
            raise ValueError('Unknown activation function: ' + str)

    @staticmethod
    def getDerivative(str):
        """
        Returns the derivative function corresponding to a given string which
        specify the activation function
        """

        if str == 'sigmoid':
            return Activation.sigmoidPrime
        elif str == 'tanh':
            return Activation.tanhPrime
        elif str == 'relu':
            return Activation.rectifiedPrime
        elif str == 'linear':
            return Activation.identityPrime
        else:
            raise ValueError('Cannot get the derivative of'
                             ' the activation function: ' + str)
