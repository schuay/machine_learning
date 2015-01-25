#!/usr/bin/env python2

import csv
import gc
import getopt
import math
import nltk
import os
import re
import sys
import time

import dataset_splitter as ds

import annealing

from nltk.classify import NaiveBayesClassifier
from nltk.classify import SklearnClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

CLASS_DEFAULT  = 'svm'
DSETS_DEFAULT  = 'annealing'
SPLIT_DEFAULT  = 'ratio75'  

DATASETS = { 'annealing': annealing.AnnealingDataset(
                    '../data/annealing/anneal.data')
#           , 'tic': tic.TICDatasetClass('../data/tic/ticdata2000.txt', None)
           }

CLASSIFIERS = { 'bayes': NaiveBayesClassifier
              , 'knn':   SklearnClassifier(KNeighborsClassifier())
              , 'svm':   SklearnClassifier(LinearSVC())
              }

SPLITTERS = { 'ratio75': ds.RatioSplitter(75)
            , 'ratiorange': ds.RatioRangeSplitter(5, 96, 5)
            , '10fold':  ds.CrossfoldSplitter(10)
            }

class Opts:
    dataset = DSETS_DEFAULT
    classifier = CLASS_DEFAULT
    splitter = SPLIT_DEFAULT
    verbose = False

options = Opts()

class ClassifierWriter:
    def __init__(self):
        self.__writer = csv.DictWriter(sys.stdout,
                                       [ "dataset"
                                       , "classifier"
                                       , "splitter"
                                       , "train_size"
                                       , "train_time"
                                       , "eval_size"
                                       , "eval_time"
                                       , "classifier_size"
                                       , "accuracy"
                                       , "class"
                                       , "class_size"
                                       , "precision"
                                       , "recall"
                                       ])

    def writeheader(self):
        self.__writer.writeheader()

    def writerow(self, train_size, train_time, eval_size, eval_time,
                 classifier_size, accuracy, cls, cls_size, precision, recall):
        self.__writer.writerow({ "dataset": options.dataset
                               , "classifier": options.classifier
                               , "splitter": options.splitter
                               , "train_size": train_size
                               , "train_time": train_time
                               , "eval_size": eval_size
                               , "eval_time": eval_time
                               , "classifier_size": classifier_size
                               , "accuracy": accuracy
                               , "class": cls
                               , "class_size": cls_size
                               , "precision": precision
                               , "recall": recall
                               })


class Classifier:
    def __init__(self, classifier, train_size, train_time):
        self.__nltk_classifier = classifier
        self.__train_size = train_size
        self.__train_time = train_time

    """Returns a classifier object trained on the given training sets."""
    @staticmethod
    def train(raw_classifier, training_sets):
        tuple_set = [ (x.features(), x.instance_class())
                      for x in training_sets.instances()
                    ]

        start = time.clock()
        trained_classifier = raw_classifier.train(tuple_set)
        elapsed = time.clock() - start

        return Classifier(trained_classifier, len(tuple_set), elapsed)

    """Evaluates the classifier with the given data sets."""
    def evaluate(self, test_sets, writer):
        class_ixs = { c: ix for ix, c in enumerate(test_sets.classes()) }

        referenceList = []
        testList = []

        start = time.clock()
        for i, inst in enumerate(test_sets.instances()):
            label = inst.instance_class()
            label_ix = class_ixs[label]
            referenceList.append(label_ix)

            predicted = self.classify(inst.features())
            predicted_ix = class_ixs[predicted]
            testList.append(predicted_ix)

        elapsed = time.clock() - start

        tuple_set = None
        gc.collect()

        accuracy = nltk.metrics.accuracy(referenceList, testList)
        return accuracy

    def classify(self, obj):
        return self.__nltk_classifier.classify(obj)

def evaluate_features(dataset, splitter, raw_classifier):
    dataset_tuples = splitter.split(dataset)

#    writer = ClassifierWriter()
#    writer.writeheader()

    accuracies = []
    for (train_set, test_set) in dataset_tuples:
        classifier = Classifier.train(raw_classifier, train_set);
        accuracies.append(classifier.evaluate(test_set, None))

    return accuracies
