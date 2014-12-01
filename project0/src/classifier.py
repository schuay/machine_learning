#!/usr/bin/env python2

import cPickle as pickle
import csv
import gc
import getopt
import math
import nltk
import os
import re
import sys
import tempfile
import time

import dataset_splitter as ds
import featureselection as fs
import transformer as tr
import tweet

import annealing
import twitter

from nltk.classify import NaiveBayesClassifier
from nltk.classify import SklearnClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

DSETS_DEFAULT  = 'twitter'
CLASS_DEFAULT  = 'svm'
NGRAM_DEFAULT  = 1
FEAT_DEFAULT   = 'aes'
SPLIT_DEFAULT  = 'ratio75'  
TRAN_DEFAULT   = 'id'

DATASETS = { 'twitter': lambda mi, fs, tr: twitter.TwitterDataset(
                    '../data/twitter/Sentiment-Analysis-Dataset.zip',
                    mi, fs, tr)
           , 'annealing': lambda mi, fs, tr: annealing.AnnealingDataset(
                    '../data/annealing/anneal.data')
           }

CLASSIFIERS = { 'bayes': NaiveBayesClassifier
              , 'knn':   SklearnClassifier(KNeighborsClassifier())
              , 'svm':   SklearnClassifier(LinearSVC())
              }

FEATURE_SELECTORS = { 'aes': lambda n: fs.AllFeatures(
                                [ fs.NGram(fs.StopWordFilter(fs.AllWords()), n)
                                , fs.Emoticons()
                                ])
                    , 'ae':  lambda n: fs.AllFeatures(
                                [fs.NGram(fs.AllWords(), n), fs.Emoticons()])
                    , 'a':   lambda n: fs.NGram(fs.AllWords(), n)
                    , 'as':  lambda n: fs.NGram(fs.StopWordFilter(fs.AllWords()), n)
                    }

SPLITTERS = { 'ratio75': ds.RatioSplitter(75)
            , 'ratiorange': ds.RatioRangeSplitter(5, 96, 1)
            , '10fold':  ds.CrossfoldSplitter(10)
            }

TRANSFORMERS = { 'id':    tr.IdentityTransformer()
               , 'url':   tr.UrlTransformer()
               , 'user':  tr.UserTransformer()
               , 'mchar': tr.MulticharTransformer()
               }

class Opts:
    dataset = DSETS_DEFAULT
    classifier = CLASS_DEFAULT
    feature_selector = FEAT_DEFAULT
    ngram = NGRAM_DEFAULT
    splitter = SPLIT_DEFAULT
    transformers = [TRAN_DEFAULT]
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

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

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

        referenceSets = [set() for x in test_sets.classes()]
        referenceList = []
        testSets = [set() for x in test_sets.classes()]
        testList = []

        start = time.clock()
        for i, inst in enumerate(test_sets.instances()):
            label = inst.instance_class()
            label_ix = class_ixs[label]
            referenceSets[label_ix].add(i)
            referenceList.append(label_ix)

            predicted = self.classify(inst.features())
            predicted_ix = class_ixs[predicted]
            testSets[predicted_ix].add(i)
            testList.append(predicted_ix)

        elapsed = time.clock() - start

        tuple_set = None
        gc.collect()

        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(self, f)
            classifier_size = os.stat(f.name).st_size

        accuracy = nltk.metrics.accuracy(referenceList, testList)

        for cl, ix in class_ixs.iteritems():
            precision = nltk.metrics.precision(referenceSets[ix], testSets[ix])
            recall = nltk.metrics.recall(referenceSets[ix], testSets[ix])
            writer.writerow(self.__train_size,
                            round(self.__train_time, 5),
                            len(test_sets.instances()),
                            round(elapsed, 5),
                            classifier_size,
                            round(accuracy, 5),
                            cl,
                            len(referenceSets[ix]),
                            precision,
                            recall)

        if options.verbose:
            try:
                print self.__nltk_classifier.show_most_informative_features(10)
            except AttributeError:
                pass # Not all classifiers provide this function.


    def classify(self, obj):
        return self.__nltk_classifier.classify(obj)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

def evaluate_features(dataset, splitter, raw_classifier):
    dataset_tuples = splitter.split(dataset)

    writer = ClassifierWriter()
    writer.writeheader()

    for (train_set, test_set) in dataset_tuples:
        classifier = Classifier.train(raw_classifier, train_set);
        train_set = None
        gc.collect()

        classifier.evaluate(test_set, writer)
        classifier, test_set = None, None
        gc.collect()

def usage():
    print("""USAGE: %s [-d dataset] [-s classifier] [-f type] [-r type] [-t type]
            -d  The dataset to use. One of 'twitter' (default), 'annealing'.
            -s  Selects the splitter. One of 'ratio75' (default), '10fold'.
            -t  Selects the classifier type. One of 'bayes', 'knn', 'svm' (default).
            -v  Verbose output.

            Twitter:
            -f  Selects the feature selector. One of %s (default = '%s').
            -g  Specifies the n for the n-gram feature selector. Can be any positive integer (default = '%s').
            -r  Enables the given transformer. Can be passed multiple times.
                One of %s (default = '%s').
                
            Annealing:
            TODO""" %
            ( sys.argv[0]
            , ", ".join(["'" + t + "'" for t in FEATURE_SELECTORS.keys()])
            , FEAT_DEFAULT
            , NGRAM_DEFAULT
            , ", ".join(["'" + t + "'" for t in TRANSFORMERS.keys()])
            , TRAN_DEFAULT
            ))
    sys.exit(1)

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], "d:f:g:hr:s:t:v")
    for o, a in opts:
        if o == "-d":
            if not a in DATASETS:
                usage()
            options.dataset = a
        elif o == "-f":
            if not a in FEATURE_SELECTORS:
                usage()
            options.feature_selector = a
        elif o == "-g":
            options.ngram = int(a)
        elif o == "-r":
            if not a in TRANSFORMERS:
                usage()
            options.transformers.append(a)
        elif o == "-s":
            if not a in SPLITTERS:
                usage()
            options.splitter = a
        elif o == "-t":
            if not a in CLASSIFIERS:
                usage()
            options.classifier = a
        elif o == "-v":
            options.verbose = True
        else:
            usage()

    dataset_ctor = DATASETS[options.dataset]
    raw_classifier = CLASSIFIERS[options.classifier]
    feature_selector = FEATURE_SELECTORS[options.feature_selector]
    ngram = options.ngram
    splitter = SPLITTERS[options.splitter]
    transformers = [TRANSFORMERS[tran] for tran in options.transformers]

    feature_selector = feature_selector(ngram)
    evaluate_features( dataset_ctor(None,
                                    feature_selector,
                                    tr.SequenceTransformer(transformers))
                     , splitter
                     , raw_classifier
                     )
