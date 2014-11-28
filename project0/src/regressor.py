#!/usr/bin/env python2

import getopt
import sys
import time

from sklearn import linear_model

import dataset_splitter as ds
import power_consumption as pc
import solar_flares as sf

DSETS_DEFAULT  = 'solar_flares'
CUTOFF_DEFAULT = 0.75

DATASETS = { 'power_consumption': lambda mi: pc.PowerConsumptionDataset(
                    '../data/power_consumption/household_power_consumption.zip',
                    mi)
           , 'solar_flares': lambda mi: sf.SolarFlaresDataset(
                    '../data/solar_flares/flare.data2')
           }

class Regressor:
    def __init__(self, clf, train_size):
        self.__clf = clf
        self.__train_size = train_size

    @staticmethod
    def train(clf, training_sets):
        xs = [ inst.x() for inst in training_sets.instances() ]
        ys = [ inst.y() for inst in training_sets.instances() ]
        return Regressor(clf.fit(xs, ys), len(xs))

    """Evaluates the classifier with the given data sets."""
    def evaluate(self, test_sets):
        for inst in test_sets.instances():
            predictions = [ int(val) for val in self.__clf.predict(inst.x()) ]
            print("ACTUAL: %s PREDICTED: %s FEATURES: %s" %
                    (inst.y(), predictions, inst.x()))

    def classify(self, obj):
        return self.__nltk_classifier.classify(obj)

def evaluate_features(dataset, splitter, raw_regressor):
    dataset_tuples = splitter.split(dataset)

    for (train_set, test_set) in dataset_tuples:
        print 'training new regressor'
        regressor = Regressor.train(raw_regressor, train_set);

        print 'testing regressor...'
        regressor.evaluate(test_set)

def usage():
    print("""USAGE: %s
            TODO""" % sys.argv[0])
    sys.exit(1)

if __name__ == '__main__':
    dataset_ctor = DATASETS[DSETS_DEFAULT]
    cutoff = CUTOFF_DEFAULT

    opts, args = getopt.getopt(sys.argv[1:], "c:d:h")
    for o, a in opts:
        if o == "-d":
            if not a in DATASETS:
                usage()
            dataset_ctor = DATASETS[a]
        elif o == "-c":
            cutoff = float(a)
        else:
            usage()

    splitter = ds.RatioSplitter(0.75)
    evaluate_features( dataset_ctor(10000)
                     , splitter
                     , linear_model.LinearRegression()
                     )
