#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator


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
                                                        learningRate=0.0005,
                                                        epochs=1000,
                                                        error='crossentropy',
                                                        network_representation=[10])

    # Train the classifiers
    print("=========================")
    print("Training..")

    #print("\nStupid Classifier has been training..")
    #myStupidClassifier.train()
    #print("Done..")

    #print("\nPerceptron has been training..")
    #myPerceptronClassifier.train()
    #print("Done..")

    print("\nLogistic Regression Classifier has been training..")
    myLogisticRegressionClassifier.train()
    print("Done..")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    #stupidPred = myStupidClassifier.evaluate()
    #perceptronPred = myPerceptronClassifier.evaluate()
    LogisticRegressionPred = myLogisticRegressionClassifier.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    #print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    #evaluator.printAccuracy(data.testSet, stupidPred)

    #print("\nResult of the Perceptron recognizer:")
    # evaluator.printComparison(data.testSet, perceptronPred)
    #evaluator.printAccuracy(data.testSet, perceptronPred)

    print("Result of the Logistic Regression:")
    evaluator.printAccuracy(data.testSet, LogisticRegressionPred)


if __name__ == '__main__':
    main()
