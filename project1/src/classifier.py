#!/usr/bin/env python2

import gc
import getopt
import math
import numpy as np
import os
import re
import sklearn
import sys
import time

from sklearn.cross_validation import KFold

class Classifier:
    def __init__(self, classifier, train_size, train_time):
        self.__classifier = classifier
        self.__train_size = train_size
        self.__train_time = train_time
        self.__test_time = None

    """Returns a classifier object trained on the given training sets."""
    @staticmethod
    def train(raw_classifier, X, y):
        start = time.clock()
        trained_classifier = raw_classifier.fit(X, y)
        elapsed = time.clock() - start

        return Classifier(trained_classifier, len(X), elapsed)

    """Evaluates the classifier with the given data sets."""
    def evaluate(self, X, y):
        predicted_y = []

        start = time.clock()
        predicted_y = [ self.classify(xs) for xs in X ]
        elapsed = time.clock() - start

        self.__test_time = elapsed

        return sklearn.metrics.accuracy_score(y, predicted_y)

    def classify(self, X):
        return self.__classifier.predict(X)

    def train_time(self):
        return self.__train_time

    # TODO: Handling of test_time is ugly, fix this.
    def test_time(self):
        return self.__test_time

def evaluate_features(dataset, raw_classifier):
    data = dataset.data()
    target = dataset.target()

    kf = KFold(len(data), n_folds = 5)

    accuracies = []
    train_times = []
    test_times = []
    for train, test in kf:
        X_train, X_test = data[train], data[test]
        y_train, y_test = target[train], target[test]

        classifier = Classifier.train(raw_classifier, X_train, y_train);
        accuracies.append(classifier.evaluate(X_test, y_test))
        train_times.append(classifier.train_time())
        test_times.append(classifier.test_time())

    # TODO: This is ugly too, should probably be an object.
    return accuracies, train_times, test_times
