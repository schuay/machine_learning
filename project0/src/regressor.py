#!/usr/bin/env python2

import getopt
import sys
import time

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import metrics

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import dataset_splitter as ds
import power_consumption as pc
import solar_flares as sf
import tic
import housing as hs

REGRE_DEFAULT  = 'linear'
DSETS_DEFAULT  = 'power_consumption'
SPLIT_DEFAULT  = 'ratio75'
MVALS_DEFAULT  = 'mean'
LIMIT_DEFAULT  = 20000
TRANS_DEFAULT  = 'scale'

DATASETS = { 'power_consumption': lambda mi: pc.PowerConsumptionDataset(
                    '../data/power_consumption/household_power_consumption.zip',
                    mi)
           , 'solar_flares': lambda mi: sf.SolarFlaresDataset(
                    '../data/solar_flares/flare.data2')
           , 'tic': lambda mi: tic.TICDataset(
                    '../data/tic/ticdata2000_f.txt', mi)
           , 'housing': lambda mi: hs.HousingDataset(
                    '../data/housing/housing.data', mi)
           }

class SingleRegressorWrapper:
    def __init__(self, raw_reg):
        self.__reg = raw_reg

    def fit(self, X, y):
        y = [ targets[0] for targets in y ]
        self.__reg.fit(X, y)
        return self

    def predict(self, X):
        return self.__reg.predict(X)

class IdTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

REGRESSORS = { 'linear': LinearRegression()
             , 'knnradius': RadiusNeighborsRegressor(radius=1)
             , 'rforest': RandomForestRegressor()
             , 'svr': SingleRegressorWrapper(SVR())
             , 'sgd': SingleRegressorWrapper(SGDRegressor(loss='huber'))
             }

SPLITTERS = { 'ratio75': ds.RatioSplitter(75)
            , 'ratiorange': ds.RatioRangeSplitter(5, 96, 5)
            , '10fold':  ds.CrossfoldSplitter(10)
            }

MISSING_VALUES_STRATEGY = { 'mean': Imputer(strategy='mean')
                          , 'median': Imputer(strategy='median')
                          }

TRANSFORMERS = { 'scale': StandardScaler()
               , 'normalize': Normalizer()
               , 'id': IdTransformer()
               }

class Opts:
    dataset = DSETS_DEFAULT
    limit = LIMIT_DEFAULT
    regressor = REGRE_DEFAULT
    splitter = SPLIT_DEFAULT
    mvals = MVALS_DEFAULT
    transformers = TRANS_DEFAULT.split(",")
    verbose = False

options = Opts()

class Regressor:
    def __init__(self, clf, mvals, train_size):
        self.__clf = clf
        self.__mvals = mvals
        self.__train_size = train_size

    @staticmethod
    def train(clf, mvals, training_sets):
        xs = [ inst.x() for inst in training_sets.instances() ]
        ys = [ inst.y() for inst in training_sets.instances() ]

        mvals = mvals.fit(ys)
        ys = mvals.transform(ys)

        return Regressor(clf.fit(xs, ys), mvals, len(xs))

    """Evaluates the classifier with the given data sets."""
    def evaluate(self, test_sets):
        xs = [ inst.x() for inst in test_sets.instances() ]
        ys = [ inst.y() for inst in test_sets.instances() ]
        ys = self.__mvals.transform(ys)

        ps = self.__clf.predict(xs)
        if not hasattr(ps[0], "__len__"):
            ps = [ [p] for p in ps ]

        if options.verbose:
            for x, y, p in zip(xs, ys, ps):
                print("ACTUAL: %s PREDICTED: %s FEATURES: %s" % (y, p, x))

        yp = zip(zip(*ys), zip(*ps))
        print map(lambda (y, p): metrics.mean_absolute_error(y,p), yp)
        print map(lambda (y, p): metrics.mean_squared_error(y,p), yp)
        print map(lambda (y, p): metrics.r2_score(y,p), yp)
        print map(lambda (y, p): metrics.explained_variance_score(y,p), yp)

    def classify(self, obj):
        return self.__nltk_classifier.classify(obj)

def evaluate_features(dataset, splitter, raw_regressor, mvals):
    dataset_tuples = splitter.split(dataset)

    for (train_set, test_set) in dataset_tuples:
        print 'training new regressor'
        regressor = Regressor.train(raw_regressor, mvals, train_set);

        print 'testing regressor...'
        regressor.evaluate(test_set)

def usage():
    print("""USAGE: %s [options]
            -d  The dataset to use. One of 'power_consumption' (default), 'solar_flares', 'tic'.
            -s  Selects the splitter. One of 'ratio75' (default), '10fold', 'ratiorange'.
            -t  Selects the regressor type. One of 'linear' (default), 'knnradius', 'rforest', 'svr', 'sgd'.
            -m  Selects the strategy for replacing missing values. One of 'mean' (default), 'median'.
            -r  Enables the given transformers, passed as a comma-separated list.
                One of %s (default = '%s').
            -l  Limit the number of rows loaded.
            -v  Verbose output.""" %
            ( sys.argv[0]
            , ", ".join(["'" + t + "'" for t in TRANSFORMERS.keys()])
            , TRANS_DEFAULT
            ))
    sys.exit(1)

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], "d:l:t:s:m:r:vh")
    for o, a in opts:
        if o == "-d":
            if not a in DATASETS:
                usage()
            options.dataset = a
        elif o == "-l":
            options.limit = None if int(a) == 0 else int(a)
        elif o == "-t":
            if not a in REGRESSORS:
                usage()
            options.regressor = a
        elif o == "-s":
            if not a in SPLITTERS:
                usage()
            options.splitter = a
        elif o == "-m":
            if not a in MISSING_VALUES_STRATEGY:
                usage()
            options.mvals = a
        elif o == "-r":
            transformers = a.split(",")
            for t in transformers:
                if not t in TRANSFORMERS:
                    usage()
            options.transformers = transformers
        elif o == "-v":
            options.verbose = True
        else:
            usage()

    pipe = list()
    for t in options.transformers:
        pipe.append((t, TRANSFORMERS[t]))
    pipe.append(('regressor', REGRESSORS[options.regressor]))

    evaluate_features( DATASETS[options.dataset](options.limit)
                     , SPLITTERS[options.splitter]
                     , Pipeline(pipe)
                     , MISSING_VALUES_STRATEGY[options.mvals]
                     )
