#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator
from sklearn.metrics import accuracy_score

import cloudpickle as pk

def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000
                      , oneHot=True, targetDigit=-1)

    myStupidClassifier = StupidRecognizer(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)

    myPerceptronClassifier = Perceptron(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,
                                        epochs=30)
    # use learningRate=0.00005 for sse if you want to see some change,
    # otherwise the batch method learnes too fast (no change after the first step)
    myLogisticRegressionClassifier = LogisticRegression(data.trainingSet,
                                                        data.validationSet,
                                                        data.testSet,
                                                        learningRate=4.,
                                                        annealing=True,
                                                        # the learning rate is divided by annealingRate * epoch
                                                        annealingRate=0.5,
                                                        momentum=1,
                                                        regularization_rate=0,
                                                        epochs=150,
                                                        error='crossentropy',
                                                        network_representation=[20],
                                                        batch_size=40)

    # Train the classifiers
    print("=========================")
    print("..")


    print("\nTraining the Feed Forward Neural Network...")
    myLogisticRegressionClassifier.train()
    print("Done!")

    # use the model
    LogisticRegressionPred = myLogisticRegressionClassifier.evaluate()

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
