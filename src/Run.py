#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.ffnn import FFNN
from report.evaluator import Evaluator
from sklearn.metrics import accuracy_score

import cloudpickle as pk

def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000
                      , oneHot=True, targetDigit="-1")

    # use learningRate=0.00005 for sse if you want to see some change,
    # otherwise the batch method learnes too fast (no change after the first step)
    myLogisticRegressionClassifier = FFNN(data.trainingSet,
                                          learningRate=2.,
                                          annealingRate=0.5,
                                          momentum=1,
                                          regularization_rate=0.01,
                                          epochs=50,
                                          error='crossentropy',
                                          network_representation=[20],
                                          batch_size=50,
                                          verbose=True)

    # Train the classifiers
    print("=========================")
    print("..")

    print("\nTraining the Feed Forward Neural Network...")
    myLogisticRegressionClassifier.train(data.trainingSet, data.validationSet)
    print("Done!")

    # use the model
    LogisticRegressionPred = myLogisticRegressionClassifier.evaluate(data.testSet.input)

    # report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the FFNN")
    accuracy = accuracy_score(data.testSet.label, LogisticRegressionPred)
    evaluator.printAccuracy(data.testSet, LogisticRegressionPred)

    print("\nSaving model...")
    model_deposit = open(myLogisticRegressionClassifier.model_descriptor + "_" +
                         str(accuracy) + "accuracy", "w")
    pk.dump(myLogisticRegressionClassifier, model_deposit)
    model_deposit.close()
    print("Done!")

if __name__ == '__main__':
    main()
