#!/usr/bin/env python2

# TODO: The spec mentions analyzing runtime (trivial for ensembles as the sum of its
# parts) and using a validation set (which sklearn documentation says is not required
# with crossfold validation).
# TODO: Can we do an ensemble of classifiers trained on random subsets of the training
# set (random forest)?

import csv
import getopt
import numpy
import sys
import ConfigParser

import d_annealing
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

DATASETS = { 'annealing': d_annealing.AnnealingDataset(
                    '../data/annealing/anneal.data')
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

        # Perform option format conversions.
        for opt in options.keys():
            convert = OPTION_CONVERSIONS.get(
                    (kind, opt), lambda o: o) # Default to identity.
            options[opt] = convert(options[opt])

        if kind == 'naive_bayes':
            return RawClassifier(GaussianNB(**options), name, options)
        elif kind == 'knn':
            return RawClassifier(KNeighborsClassifier(**options),
                                 name, options)
        elif kind == 'svm':
            return RawClassifier(LinearSVC(**options),
                                 name, options)
        elif kind == 'tree':
            return RawClassifier(DecisionTreeClassifier(**options),
                                 name, options)
        elif kind == 'extra_tree':
            return RawClassifier(ExtraTreeClassifier(**options),
                                 name, options)
        else:
            assert False, "%s: invalid 'kind' attribute." % name

# TODO: Rename to something more appropriate, as this is more of a classifier
# wrapper than a raw classifier.
class RawClassifier:
    def __init__(self, raw_classifier, name, options):
        self.raw_classifier = raw_classifier
        self.name = name
        self.options = options
        self.accuracies = None

    def mean_accuracy(self):
        assert self.accuracies is not None
        return numpy.mean(self.accuracies)

    def std_deviation(self):
        assert self.accuracies is not None
        return numpy.std(self.accuracies)

    @staticmethod
    def evaluate(raw_classifier, dataset):
        verbose("Evaluating classifier '%s': %s" %
                (raw_classifier.name, raw_classifier.options))

        raw_classifier.accuracies = cl.evaluate_features(
                dataset, raw_classifier.raw_classifier)

        verbose("Mean accuracy: %s" % round(raw_classifier.mean_accuracy(), 4))

def load_classifiers(config_file):
    """Loads classifiers as specified in config_file and returns them as a list."""
    cp = ConfigParser.RawConfigParser()
    cp.optionxform = str # Preserve case of option names.

    with open(config_file) as f:
        cp.readfp(f)

    classifiers = []
    for cl_instance in cp.sections():
        cl_options = dict(cp.items(cl_instance))
        classifiers.append(RawClassifierFactory.new(cl_instance, cl_options))

    return classifiers

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
                                       ])

    def writeheader(self):
        self.__writer.writeheader()

    def writerow(self, dataset, raw_classifier):
        self.__writer.writerow({ "dataset": dataset.name()
                               , "classifier": raw_classifier.name
                               , "mean_accuracy": raw_classifier.mean_accuracy()
                               , "std_deviation": raw_classifier.std_deviation()
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

    # Load all classifiers from the configuration file.
    classifiers = load_classifiers(options.config_file)

    # Initially evaluate all loaded classifiers and store accuracies.
    dataset = DATASETS[options.dataset]
    for raw_classifier in classifiers:
        RawClassifier.evaluate(raw_classifier, dataset)

    # Evaluate the simple and weighted ensemble classifiers.
    simple_ensemble = RawClassifier(
            EnsembleClassifier(classifiers, simple_majority),
            "ensemble",
            { 'majority_function': 'simple_majority' }
            )

    weighted_ensemble = RawClassifier(
            EnsembleClassifier(classifiers, weighted_majority),
            "ensemble",
            { 'majority_function': 'weighted_majority' }
            )

    RawClassifier.evaluate(simple_ensemble, dataset)
    RawClassifier.evaluate(weighted_ensemble, dataset)

    # Output results.
    all_classifiers = list(classifiers)
    all_classifiers.extend([simple_ensemble, weighted_ensemble])

    writer = ClassifierWriter()
    writer.writeheader()
    for raw_classifier in all_classifiers:
        writer.writerow(dataset, raw_classifier)
