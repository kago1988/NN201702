# -*- coding: utf-8 -*-

import numpy as np

class DataSet(object):
    """
    Representing train, valid or test sets

    Parameters
    ----------
    data : list
    oneHot : bool
        If this flag is set, then all labels which are not `targetDigit` will
        be transformed to False and `targetDigit` bill be transformed to True.
    targetDigit : string
        Label of the dataset, e.g. '7'.

    Attributes
    ----------
    input : list
    label : list
        A labels for the data given in `input`.
    oneHot : bool
    targetDigit : string
    """

    def __init__(self, data, oneHot=True, targetDigit='7'):

        # The label of the digits is always the first fields
        # Doing normalization
        self.input = 1.0*data[:, 1:]/255
        self.label = data[:, 0]
        self.oneHot = oneHot
        self.targetDigit = targetDigit

        # Transform all labels which is not the targetDigit to False,
        # The label of targetDigit will be True,
        if oneHot and targetDigit != -1:
            self.label = list(map(lambda a: 1
                            if str(a) == targetDigit else 0,
                            self.label))
        elif oneHot and targetDigit == -1:
            a = list(map(lambda a: int(a), self.label))
            b = np.zeros((len(a), 10))
            b[np.arange(len(a)), np.array(a)] = 1
            self.label = b
        else:
            raise ValueError("Could not infer label format. Select either oneHot "
                             "with a targetDigit between 0 and 9 (for binary class classification) "
                             "or oneHot with targetDigit -1 (for multiclass classification).")

    def __iter__(self):
        return self.input.__iter__()
