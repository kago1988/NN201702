#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.ffnn import FFNN
from report.evaluator import Evaluator
from sklearn.metrics import accuracy_score

import cloudpickle as pk


def main():
    # mnist_seven has 5000 data-points: use for development and debugging
    # mnist_train has 60000 data-points: use for training
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000
                      , oneHot=True, targetDigit="-1")

    myLogisticRegressionClassifier = FFNN(data.trainingSet,
                                          learningRate=0.05,
                                          momentum=1,
                                          regularization_rate=0.03,
                                          epochs=50,
                                          error='crossentropy',
                                          network_representation=[15],
                                          batch_size=100,
                                          verbose=True,
                                          normalized=True)

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
