#!/usr/bin/env python2

import gc
import math
import cPickle as pickle
import re
import time

import featureselection as fs
import transformer as tr
import tweet

from nltk.classify import NaiveBayesClassifier
from nltk.classify import SklearnClassifier
from nltk.classify.util import apply_features
from sklearn.svm import LinearSVC

NEG = 0
POS = 1

CLASSIFIERS = { 'bayes': NaiveBayesClassifier
              , 'svm':   SklearnClassifier(LinearSVC())
              }

class Classifier:
    def __init__(self, classifier, feature_selection, transformer, train_size):
        self.__nltk_classifier = classifier
        self.__fs = feature_selection
        self.__tr = transformer
        self.__train_size = train_size

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    """Takes a dictionary with keys: POSITIVE/NEGATIVE, values: list of
    individual tweets. Returns a classifier object trained on the given training sets."""
    @staticmethod
    def train(raw_classifier, training_sets, feature_selection, transformer):
        training = []

        # Since we have a rather large amount of training data, build features
        # lazily to avoid running out of memory.
        tuple_set = [(transformer.transform(x), cl)
                             for cl in [POS, NEG]
                             for x in training_sets[cl]]
        train_set = apply_features(feature_selection.select_features, tuple_set)

        return Classifier(raw_classifier.train(train_set), feature_selection,
                transformer, len(tuple_set))

    """Evaluates the classifier with the given data sets."""
    def evaluate(self, test_sets):
        tuple_set = [(self.__tr.transform(x), cl)
                             for cl in [POS, NEG]
                             for x in test_sets[cl]]
        referenceSets = [set() for x in [POS, NEG]]
        referenceList = []
        testSets = [set() for x in [POS, NEG]]
        testList = []
        start = time.clock()
        for i, (t, label) in enumerate(tuple_set):
            referenceSets[label].add(i)
            referenceList.append(label)
            predicted = self.classify(t)
            testSets[predicted].add(i)
            testList.append(predicted)
        elapsed = time.clock() - start

        tuple_set = None
        gc.collect()

        print 'train on %d instances, test on %d instances' % (self.__train_size,
                sum(map(len, test_sets)))
        print 'classified evaluation set in %f seconds' % elapsed
        print 'accuracy:', nltk.metrics.accuracy(referenceList, testList)
        print 'pos precision:', nltk.metrics.precision(referenceSets[POS], testSets[POS])
        print 'pos recall:', nltk.metrics.recall(referenceSets[POS], testSets[POS])
        print 'neg precision:', nltk.metrics.precision(referenceSets[NEG], testSets[NEG])
        print 'neg recall:', nltk.metrics.recall(referenceSets[NEG], testSets[NEG])

        try:
            print self.__nltk_classifier.show_most_informative_features(10)
        except AttributeError:
            pass # Not all classifiers provide this function.


    def classify(self, obj):
        transformed = self.__tr.transform(obj)
        features = self.__fs.select_features(transformed)
        return self.__nltk_classifier.classify(features)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

import getopt
import nltk
import re
import sys

def prefilter(tweets):
    PATTERN_SPAM1 = re.compile("Get 100 followers a day")
    PATTERN_SPAM2 = re.compile("I highly recommends you join www.m2e.asia")
    PATTERN_SPAM3 = re.compile("Banksyart2.*posting there since having probs")

    FILTERS = [ lambda t: not PATTERN_SPAM1.search(t)
              , lambda t: not PATTERN_SPAM2.search(t)
              , lambda t: not PATTERN_SPAM3.search(t)
              ]

    return filter(lambda t: all([f(t[tweet.TEXT]) for f in FILTERS]), tweets)

def to_tweets(lines):
    """Turns a list of tweet texts into a list of tweet dict objects."""
    return [{tweet.TEXT: t} for t in lines]

def evaluate_features(positive, negative, load, save, cutoff,
                      raw_classifier, feature_selector, transformer):
    with open(positive, 'r') as f:
        posTweets = prefilter(to_tweets(re.split(r'\n', f.read())))
    with open(negative, 'r') as f:
        negTweets = prefilter(to_tweets(re.split(r'\n', f.read())))
 
    # Selects cutoff of the features to be used for training and (1 - cutoff)
    # to be used for testing.
    posCutoff = int(math.floor(len(posTweets)*cutoff))
    negCutoff = int(math.floor(len(negTweets)*cutoff))

    if load:
        print 'loading classifier \'%s\'' % load
        classifier = Classifier.load(load)

    else:
        print 'training new classifier'

        trainSets = [list() for x in [POS, NEG]]
        trainSets[POS] = posTweets[:posCutoff]
        trainSets[NEG] = negTweets[:negCutoff]

        classifier = Classifier.train(raw_classifier, trainSets,
                feature_selector, transformer);

        trainSets = None
        gc.collect()

    trainSets = [list() for x in [POS, NEG]]
    trainSets[POS] = posTweets[posCutoff:]
    trainSets[NEG] = negTweets[negCutoff:]
    posTweets, negTweets = None, None # Free some space.
    gc.collect()

    if save:
        print 'saving classifier as \'%s\'' % save

        classifier.save(save)

    print 'testing classifier...'

    classifier.evaluate(trainSets)

NGRAM_DEFAULT = 1

FEAT_DEFAULT = 'aes'
FEATURE_SELECTORS = { 'aes': lambda n: fs.AllFeatures([fs.NGram(fs.StopWordFilter(fs.AllWords()), n), fs.Emoticons()])
                    , 'ae':  lambda n: fs.AllFeatures([fs.NGram(fs.AllWords(), n), fs.Emoticons()])
                    , 'a':   lambda n: fs.NGram(fs.AllWords(), n)
                    , 'as':  lambda n: fs.NGram(fs.StopWordFilter(fs.AllWords()), n)
                    }

TRAN_DEFAULT = 'id'
TRANSFORMERS = { 'id':    tr.IdentityTransformer()
               , 'url':   tr.UrlTransformer()
               , 'user':  tr.UserTransformer()
               , 'mchar': tr.MulticharTransformer()
               }

def usage():
    print("""USAGE: %s [-p positive_tweets] [-n negative_tweets] [-s classifier]
                       [-l classifier] [-c cutoff] [-f type] [-r type] [-t type]
            -p  Text file containing one positive tweet per line.
            -n  Text file containing one negative tweet per line.
            -s  Saves the classifier to the specified file.
            -l  Loads the classifier from the specified file.
            -c  Specifies the percentage of training tweets (default = 0.75).
            -f  Selects the feature selector. One of %s (default = '%s').
            -g  Specifies the n for the n-gram feature selector. Can be any positive integer (default = '%s').
            -r  Enables the given transformer. Can be passed multiple times.
                One of %s (default = '%s').
            -t  Selects the classifier type. One of 'bayes', 'svm' (default).""" %
            ( sys.argv[0]
            , ", ".join(["'" + t + "'" for t in FEATURE_SELECTORS.keys()])
            , FEAT_DEFAULT
            , NGRAM_DEFAULT
            , ", ".join(["'" + t + "'" for t in TRANSFORMERS.keys()])
            , TRAN_DEFAULT
            ))
    sys.exit(1)

# TODO: Since we now need to download nltk stopwords, mention this in the readme
# or implement automatic downloading into a local dir within this script.

if __name__ == '__main__':
    classifier_save = None
    classifier_load = None

    positive_file = 'sentiment.pos'
    negative_file = 'sentiment.neg'
    cutoff = 0.75
    raw_classifier = CLASSIFIERS['svm']
    feature_selector = FEATURE_SELECTORS[FEAT_DEFAULT]
    ngram = NGRAM_DEFAULT
    transformers = [TRANSFORMERS[TRAN_DEFAULT]]

    opts, args = getopt.getopt(sys.argv[1:], "hc:s:l:p:n:c:t:f:g:r:")
    for o, a in opts:
        if o == "-s":
            classifier_save = a
        elif o == "-l":
            classifier_load = a
        elif o == "-p":
            positive_file = a
        elif o == "-n":
            negative_file = a
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

    feature_selector = feature_selector(ngram)
    evaluate_features( positive_file
                     , negative_file
                     , classifier_load
                     , classifier_save
                     , cutoff
                     , raw_classifier
                     , feature_selector
                     , tr.SequenceTransformer(transformers)
                     )
