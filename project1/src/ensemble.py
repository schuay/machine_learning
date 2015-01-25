#!/usr/bin/env python2

# TODO: A module to read used classifiers and their parameters from a config file
#       (command line args would be too long and messy). There must be a ready-made module
#       around that can do this.
# TODO: Evaluate accuracy on all classifiers using crossfold validation. Store results,
#       including data to write to CSV later on.
# TODO: Generate ensemble classifier (weighted & std) and evaluate.
# TODO: Print results to stdout.

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

def usage():
    print("""USAGE: %s [options]
            -c  The configuration file to load.`
            -d  The dataset to use. One of %s.
            -v  Verbose output.

            Config file syntax is:

            [classifier_instance_name]
            type = the_classifier_type
            option = value
            [...]""" %
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

class RawClassifier:
    def __init__(self, raw_classifier, name, options):
        self.raw_classifier = raw_classifier
        self.name = name
        self.options = options

def load_classifiers(config_file):
    """Loads classifiers as specified in config_file and returns them as a list."""
    cp = ConfigParser.RawConfigParser()
    cp.optionxform = str # Preserve case of option names.

    with open(config_file) as f:
        cp.readfp(f)

    classifiers = []
    for cl_instance in cp.sections():
        cl_options = dict(cp.items(cl_instance))

        # TODO: Figure out if string -> {int,float} conversions are done implicitly
        # (e.g. for svm: C = '1.0').
        classifiers.append(RawClassifierFactory.new(cl_instance, cl_options))

    return classifiers


def verbose(message):
    if options.verbose:
        print message

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

    classifiers = load_classifiers(options.config_file)

    splitter = ds.CrossfoldSplitter(5)
    for raw_classifier in classifiers:
        verbose("Evaluating classifier '%s': %s" %
                (raw_classifier.name, raw_classifier.options))
        dataset = cl.DATASETS[options.dataset]

        # TODO: CSV output still relies on classifier.py global options.
        # We will need to alter this to output accuracy only (precision & recall
        # mess up the CSV too much and we won't be able to present this well),
        # to properly insert only a single header row at the beginning of the file,
        # and to return the resulting averaged accuracy. It might be best to move
        # CSV writing out here and simply return a result list.
        cl.evaluate_features( dataset
                            , splitter
                            , raw_classifier.raw_classifier
                            )
