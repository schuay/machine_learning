#!/usr/bin/env python2

import csv
import getopt
import numpy
import random
import sys
import ConfigParser

import d_annealing
import d_sklearn
import classifier as cl

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

class Opts:
    config_file = "conf/default.conf"
    dataset = "annealing"
    verbose = False

options = Opts()

# Add float and integer options for classifiers here.
OPTION_CONVERSIONS = { ('svm', 'C'): lambda o: float(o)
                     , ('knn', 'n_neighbors'): lambda o: int(o)
                     }

DATASETS = { 'annealing': lambda: d_annealing.AnnealingDataset(
                    '../data/annealing/anneal.data')
           , 'iris': lambda: d_sklearn.IrisDataset()
           , 'digits': lambda: d_sklearn.DigitsDataset()
           , 'random2': lambda: d_sklearn.Random2Dataset()
           , 'random10': lambda: d_sklearn.Random10Dataset()
           , 'hastie': lambda: d_sklearn.HastieDataset()
           , 'breast-cancer_scale': lambda: d_sklearn.LibSVMDataset(
                    '../data/libsvm/breast-cancer_scale')
           , 'heart_scale': lambda: d_sklearn.LibSVMDataset(
                    '../data/libsvm/heart_scale')
           , 'connect-4': lambda: d_sklearn.LibSVMDataset(
                    '../data/libsvm/connect-4')
           , 'vehicle.scale': lambda: d_sklearn.LibSVMDataset(
                    '../data/libsvm/vehicle.scale')
           , 'vowel.scale': lambda: d_sklearn.LibSVMDataset(
                    '../data/libsvm/vowel.scale')
           }

def usage():
    print("""USAGE: %s [options]
            -c  The configuration file to load.`
            -d  The dataset to use. One of %s.
            -v  Verbose output.""" %
            ( sys.argv[0]
            , DATASETS.keys()
            )
         )
    sys.exit(1)

class RawClassifierFactory:
    @staticmethod
    def new(name, options):
        assert 'kind' in options, "%s: missing 'kind' attribute." % name

        kind = options.pop('kind')
        subset_ratio = options.pop('training_subset_ratio', None)
        subset_seed  = options.pop('training_subset_seed', None)

        # Perform option format conversions.
        for opt in options.keys():
            convert = OPTION_CONVERSIONS.get(
                    (kind, opt), lambda o: o) # Default to identity.
            options[opt] = convert(options[opt])

        if kind == 'naive_bayes':
            c = RawClassifier(GaussianNB(**options), name, options)
        elif kind == 'knn':
            c = RawClassifier(KNeighborsClassifier(**options),
                                 name, options)
        elif kind == 'svm':
            c = RawClassifier(LinearSVC(**options),
                                 name, options)
        elif kind == 'tree':
            c = RawClassifier(DecisionTreeClassifier(**options),
                                 name, options)
        elif kind == 'extra_tree':
            c = RawClassifier(ExtraTreeClassifier(**options),
                                 name, options)
        else:
            assert False, "%s: invalid 'kind' attribute." % name

        if subset_ratio is not None:
            c = RawClassifier( SubsetClassifier( float(subset_ratio)
                                               , int(subset_seed)
                                               , c.raw_classifier
                                               )
                             , c.name
                             , c.options
                             )

        return c

class SubsetClassifier:
    def __init__(self, ratio, seed, inner_classifier):
        self.__ratio = ratio
        self.__seed = seed
        self.__inner_classifier = inner_classifier

    def fit(self, X, y):
        rows, cols = X.shape

        random.seed(self.__seed)
        m = [ 0 if random.random() <= self.__ratio else 1 for _ in xrange(rows) ]

        my = numpy.ma.compressed(numpy.ma.masked_array(y, mask = m))
        mX = numpy.ma.compress_rows(
                numpy.ma.masked_array(X, mask = [ [z] * cols for z in m ]))

        self.__inner_classifier.fit(mX, my)

        return self.__inner_classifier

# TODO: Rename to something more appropriate, as this is more of a classifier
# wrapper than a raw classifier.
class RawClassifier:
    def __init__(self, raw_classifier, name, options):
        self.raw_classifier = raw_classifier
        self.name = name
        self.options = options
        self.accuracies = None
        self.train_times = None
        self.test_times = None

    def mean_accuracy(self):
        assert self.accuracies is not None
        return numpy.mean(self.accuracies)

    def std_deviation(self):
        assert self.accuracies is not None
        return numpy.std(self.accuracies)

    def mean_train_time(self):
        assert self.train_times is not None
        return numpy.mean(self.train_times)

    def mean_test_time(self):
        assert self.test_times is not None
        return numpy.mean(self.test_times)

    @staticmethod
    def evaluate(raw_classifier, dataset):
        verbose("Evaluating classifier '%s': %s" %
                (raw_classifier.name, raw_classifier.options))

        acc, trt, tst = cl.evaluate_features(dataset, raw_classifier.raw_classifier)

        raw_classifier.accuracies = acc
        raw_classifier.train_times = trt
        raw_classifier.test_times = tst

        verbose("Mean accuracy: %s, Std Deviation: %s, "
                 "Mean training time %s, Mean testing time %s" % (
            round(raw_classifier.mean_accuracy(), 4),
            round(raw_classifier.std_deviation(), 4),
            round(raw_classifier.mean_train_time(), 4),
            round(raw_classifier.mean_test_time(), 4)))

class ConfigOptions:
    def __init__(self, config_file):
        cp = ConfigParser.RawConfigParser()
        cp.optionxform = str # Preserve case of option names.

        with open(config_file) as f:
            cp.readfp(f)

        classifiers = dict()
        ensembles = None
        for cl_instance in cp.sections():
            if (cl_instance == 'ensembles'):
                ensembles = dict(cp.items('ensembles'))
                continue
            cl_options = dict(cp.items(cl_instance))
            classifiers[cl_instance] = RawClassifierFactory.new(cl_instance, cl_options)

        if not ensembles:
            raise ConfigParser.NoSectionError('No \'ensembles\' section specified.')

        self.__ensembles = dict()
        for e, cs in ensembles.iteritems():
            self.__ensembles[e] = [ classifiers[s.strip()] for s in cs.split(',') ]

        self.__classifiers = classifiers.values()

    def ensembles(self):
        return self.__ensembles

    def classifiers(self):
        return self.__classifiers

class EnsembleClassifier:
    """An ensemble classifier combines the results of several different classifiers."""
    def __init__(self, raw_classifiers, majority_function):
        self.__raw_classifiers = raw_classifiers
        self.__majority_function = majority_function

    def fit(self, X, y):
        self.__trained_classifiers = list()
        for c in self.__raw_classifiers:
            classifier = c.raw_classifier.fit(X, y)
            self.__trained_classifiers.append(classifier)

        return self

    def predict(self, X):
        predictions = dict()
        for (trained, raw) in zip(self.__trained_classifiers, self.__raw_classifiers):
            predictions[raw] = trained.predict(X)[0]

        return self.__majority_function(predictions)

def simple_majority(predictions):
    counts = dict()
    for p in predictions.itervalues():
        counts[p] = counts[p] + 1 if p in counts else 0

    winner = counts.iterkeys().next()
    for p, c in counts.iteritems():
        if (c > counts[winner]):
            winner = p

    return winner

def weighted_majority(predictions):
    counts = dict()
    for raw, p in predictions.iteritems():
        weight = raw.mean_accuracy()
        counts[p] = counts[p] + weight if p in counts else 0

    winner = counts.iterkeys().next()
    for p, c in counts.iteritems():
        if (c > counts[winner]):
            winner = p

    return winner

def verbose(message):
    if options.verbose:
        print message

class ClassifierWriter:
    def __init__(self):
        self.__writer = csv.DictWriter(sys.stdout,
                                       [ "dataset"
                                       , "classifier"
                                       , "mean_accuracy"
                                       , "std_deviation"
                                       , "mean_train_time"
                                       , "mean_test_time"
                                       ])

    def writeheader(self):
        self.__writer.writeheader()

    def writerow(self, dataset, raw_classifier):
        self.__writer.writerow({ "dataset": dataset.name()
                               , "classifier": raw_classifier.name
                               , "mean_accuracy": raw_classifier.mean_accuracy()
                               , "std_deviation": raw_classifier.std_deviation()
                               , "mean_train_time": raw_classifier.mean_train_time()
                               , "mean_test_time": raw_classifier.mean_test_time()
                               })

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], "c:d:hv")
    for o, a in opts:
        if o == "-c":
            options.config_file = a
        elif o == "-d":
            if not a in DATASETS:
                usage()
            options.dataset = a
        elif o == "-v":
            options.verbose = True
        else:
            usage()

    # Parse the configuration file.
    copts = ConfigOptions(options.config_file)

    # Load all classifiers from the configuration file.
    classifiers = copts.classifiers()

    # Initially evaluate all loaded classifiers and store accuracies.
    dataset = DATASETS[options.dataset]()
    for raw_classifier in classifiers:
        RawClassifier.evaluate(raw_classifier, dataset)

    all_classifiers = list(classifiers)

    for e, cs in copts.ensembles().iteritems():
        # Evaluate the simple and weighted ensemble classifiers.
        simple_ensemble = RawClassifier(
                EnsembleClassifier(cs, simple_majority),
                "%s_simple_ensemble" % e,
                { 'majority_function': 'simple_majority' }
                )

        weighted_ensemble = RawClassifier(
                EnsembleClassifier(cs, weighted_majority),
                "%s_weighted_ensemble" % e,
                { 'majority_function': 'weighted_majority' }
                )

        RawClassifier.evaluate(simple_ensemble, dataset)
        RawClassifier.evaluate(weighted_ensemble, dataset)

        all_classifiers.extend([simple_ensemble, weighted_ensemble])

    # Output results.
    writer = ClassifierWriter()
    writer.writeheader()
    for raw_classifier in all_classifiers:
        writer.writerow(dataset, raw_classifier)
