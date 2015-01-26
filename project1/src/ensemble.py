#!/usr/bin/env python2

import csv
import getopt
import sys
import ConfigParser

import classifier as cl
import dataset_splitter as ds

from nltk.classify import NaiveBayesClassifier
from nltk.classify import SklearnClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

class Opts:
    config_file = "conf/default.conf"
    dataset = "annealing"
    verbose = False

options = Opts()

# Add float and integer options for classifiers here.
OPTION_CONVERSIONS = { ('svm', 'C'): lambda o: float(o)
                     , ('knn', 'n_neighbors'): lambda o: int(o)
                     }

def usage():
    print("""USAGE: %s [options]
            -c  The configuration file to load.`
            -d  The dataset to use. One of %s.
            -v  Verbose output.""" %
            ( sys.argv[0]
            , cl.DATASETS.keys()
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
            convert = OPTION_CONVERSIONS.get((kind, opt)
                                           , lambda o: o) # Default to identity.
            options[opt] = convert(options[opt])

        if kind == 'naive_bayes':
            return RawClassifier(NaiveBayesClassifier, name, options)
        elif kind == 'knn':
            return RawClassifier(SklearnClassifier(KNeighborsClassifier(**options)),
                                 name, options)
        elif kind == 'svm':
            return RawClassifier(SklearnClassifier(LinearSVC(**options)),
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
        return sum(self.accuracies) / len(self.accuracies)

    def std_deviation(self):
        assert self.accuracies is not None
        return 0.0 # TODO

    @staticmethod
    def evaluate(raw_classifier, splitter, dataset):
        verbose("Evaluating classifier '%s': %s" %
                (raw_classifier.name, raw_classifier.options))

        raw_classifier.accuracies = cl.evaluate_features(
                dataset, splitter, raw_classifier.raw_classifier)

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

    def train(self, tuple_set):
        self.__trained_classifiers = list()
        for c in self.__raw_classifiers:
            classifier = c.raw_classifier.train(tuple_set)
            self.__trained_classifiers.append(classifier)

        return self

    def classify(self, instance):
        predictions = dict()
        for (trained, raw) in zip(self.__trained_classifiers, self.__raw_classifiers):
            predictions[raw] = trained.classify(instance)

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
                                       , "splitter"
                                       , "mean_accuracy"
                                       , "std_deviation"
                                       ])

    def writeheader(self):
        self.__writer.writeheader()

    def writerow(self, dataset, raw_classifier, splitter):
        self.__writer.writerow({ "dataset": dataset.name()
                               , "classifier": raw_classifier.name
                               , "splitter": splitter.name()
                               , "mean_accuracy": raw_classifier.mean_accuracy()
                               , "std_deviation": raw_classifier.std_deviation()
                               })

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], "c:d:hv")
    for o, a in opts:
        if o == "-c":
            options.config_file = a
        elif o == "-d":
            if not a in cl.DATASETS:
                usage()
            options.dataset = a
        elif o == "-v":
            options.verbose = True
        else:
            usage()

    # Load all classifiers from the configuration file.
    classifiers = load_classifiers(options.config_file)

    # Initially evaluate all loaded classifiers and store accuracies.
    dataset = cl.DATASETS[options.dataset]
    splitter = ds.CrossfoldSplitter(5)
    for raw_classifier in classifiers:
        RawClassifier.evaluate(raw_classifier, splitter, dataset)

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

    RawClassifier.evaluate(simple_ensemble, splitter, dataset)
    RawClassifier.evaluate(weighted_ensemble, splitter, dataset)

    # Output results.
    all_classifiers = list(classifiers)
    all_classifiers.extend([simple_ensemble, weighted_ensemble])

    writer = ClassifierWriter()
    writer.writeheader()
    for raw_classifier in all_classifiers:
        writer.writerow(dataset, raw_classifier, splitter)
