#!/usr/bin/env python2

import cPickle as pickle
import gc
import getopt
import math
import nltk
import re
import sys
import time

import dataset_splitter as ds
import featureselection as fs
import transformer as tr
import tweet

import annealing
import twitter

from nltk.classify import NaiveBayesClassifier
from nltk.classify import SklearnClassifier
from sklearn.svm import LinearSVC

DSETS_DEFAULT  = 'twitter'
CUTOFF_DEFAULT = 0.75
CLASS_DEFAULT  = 'svm'
NGRAM_DEFAULT  = 1
FEAT_DEFAULT   = 'aes'
TRAN_DEFAULT   = 'id'

DATASETS = { 'twitter': lambda mi, fs, tr: twitter.TwitterDataset(
                    '../data/twitter/Sentiment-Analysis-Dataset.zip',
                    mi, fs, tr)
           , 'annealing': lambda mi, fs, tr: annealing.AnnealingDataset(
                    '../data/annealing/anneal.data')
           }

CLASSIFIERS = { 'bayes': NaiveBayesClassifier
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

TRANSFORMERS = { 'id':    tr.IdentityTransformer()
               , 'url':   tr.UrlTransformer()
               , 'user':  tr.UserTransformer()
               , 'mchar': tr.MulticharTransformer()
               }

class Classifier:
    def __init__(self, classifier, train_size):
        self.__nltk_classifier = classifier
        self.__train_size = train_size

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    """Takes a dictionary with keys: POSITIVE/NEGATIVE, values: list of
    individual tweets. Returns a classifier object trained on the given training sets."""
    @staticmethod
    def train(raw_classifier, training_sets):
        tuple_set = [ (x.features(), x.instance_class())
                      for x in training_sets.instances()
                    ]
        return Classifier(raw_classifier.train(tuple_set), len(tuple_set))

    """Evaluates the classifier with the given data sets."""
    def evaluate(self, test_sets):
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

            print("ACTUAL: %s, PREDICTED: %s FEATURES: %s" % (label, predicted, inst.features()))
        elapsed = time.clock() - start

        tuple_set = None
        gc.collect()

        print 'train on %d instances, test on %d instances' % (self.__train_size,
                len(test_sets.instances()))
        print 'classified evaluation set in %f seconds' % elapsed
        print 'accuracy:', nltk.metrics.accuracy(referenceList, testList)

        for cl, ix in class_ixs.iteritems():
            precision = nltk.metrics.precision(referenceSets[ix], testSets[ix])
            recall = nltk.metrics.recall(referenceSets[ix], testSets[ix])
            print '%s precision: %s' % (cl, precision)
            print '%s recall: %s' % (cl, recall) 

        if test_sets.kind() == twitter.KIND_TWITTER:
            # TODO: Possibly extract this to function defined in dataset class.
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

    for (train_set, test_set) in dataset_tuples:
        print 'training new classifier'
        classifier = Classifier.train(raw_classifier, train_set);

        print 'testing classifier...'
        classifier.evaluate(test_set)

def usage():
    print("""USAGE: %s [-d dataset] [-s classifier] [-f type] [-r type] [-t type]
            -d  The dataset to use. One of 'twitter' (default), 'annealing'.
            -t  Selects the classifier type. One of 'bayes', 'svm' (default).

            Twitter:
            -c  Specifies the percentage of training tweets (default = 0.75).
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
    dataset_ctor = DATASETS[DSETS_DEFAULT]
    cutoff = CUTOFF_DEFAULT
    raw_classifier = CLASSIFIERS[CLASS_DEFAULT]
    feature_selector = FEATURE_SELECTORS[FEAT_DEFAULT]
    ngram = NGRAM_DEFAULT
    transformers = [TRANSFORMERS[TRAN_DEFAULT]]

    opts, args = getopt.getopt(sys.argv[1:], "c:d:f:g:hr:t:")
    for o, a in opts:
        if o == "-d":
            if not a in DATASETS:
                usage()
            dataset_ctor = DATASETS[a]
        elif o == "-c":
            cutoff = float(a)
        elif o == "-t":
            if not a in CLASSIFIERS:
                usage()
            raw_classifier = CLASSIFIERS[a]
        elif o == "-f":
            if not a in FEATURE_SELECTORS:
                usage()
            feature_selector = FEATURE_SELECTORS[a]
        elif o == "-g":
            ngram = int(a)
        elif o == "-r":
            if not a in TRANSFORMERS:
                usage()
            transformers.append(TRANSFORMERS[a])
        else:
            usage()

    splitter = ds.RatioSplitter(0.75)
    feature_selector = feature_selector(ngram)
    evaluate_features( dataset_ctor(10000,
                                    feature_selector,
                                    tr.SequenceTransformer(transformers))
                     , splitter
                     , raw_classifier
                     )
