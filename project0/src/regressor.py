#!/usr/bin/env python2

import getopt
import sys
import time

from sklearn.linear_model import LinearRegression

import dataset_splitter as ds
import power_consumption as pc
import solar_flares as sf

REGRE_DEFAULT  = 'linear'
DSETS_DEFAULT  = 'power_consumption'
SPLIT_DEFAULT  = 'ratio75'
LIMIT_DEFAULT  = 50000

DATASETS = { 'power_consumption': lambda mi: pc.PowerConsumptionDataset(
                    '../data/power_consumption/household_power_consumption.zip',
                    mi)
           , 'solar_flares': lambda mi: sf.SolarFlaresDataset(
                    '../data/solar_flares/flare.data2')
           }

REGRESSORS = { 'linear': LinearRegression()
             }

SPLITTERS = { 'ratio75': ds.RatioSplitter(75)
            , 'ratiorange': ds.RatioRangeSplitter(5, 96, 5)
            , '10fold':  ds.CrossfoldSplitter(10)
            }

class Opts:
    dataset = DSETS_DEFAULT
    limit = LIMIT_DEFAULT
    regressor = REGRE_DEFAULT
    splitter = SPLIT_DEFAULT
    verbose = False

options = Opts()

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
        squares = list()
        for inst in test_sets.instances():
            predictions = [ val for val in self.__clf.predict(inst.x()) ]
            print("ACTUAL: %s PREDICTED: %s FEATURES: %s" %
                    (inst.y(), predictions, inst.x()))
            squares.append([(y - py)**2 for y,py in zip(inst.y(), predictions)])
        print map(lambda l: sum(l)/len(l), zip(*squares))

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
    print("""USAGE: %s [options]
            -d  The dataset to use. One of 'power_consumption' (default), 'solar_flares'.
            -s  Selects the splitter. One of 'ratio75' (default), '10fold', 'ratiorange'.
            -t  Selects the regressor type. One of 'linear' (default).
            -l  Limit the number of rows loaded.
            -v  Verbose output.""" %
            ( sys.argv[0]
            ))
    sys.exit(1)

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], "d:l:t:s:vh")
    for o, a in opts:
        if o == "-d":
            if not a in DATASETS:
                usage()
            options.dataset = a
        elif o == "-l":
            options.limit = int(a)
        elif o == "-t":
            if not a in REGRESSORS:
                usage()
            options.regressor = a
        elif o == "-s":
            if not a in SPLITTERS:
                usage()
            options.splitter = a
        elif o == "-v":
            options.verbose = True
        else:
            usage()

    evaluate_features( DATASETS[options.dataset](options.limit)
                     , SPLITTERS[options.splitter]
                     , REGRESSORS[options.regressor]
                     )
