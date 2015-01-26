# Datasets loaded using sklearn.datasets utilities.

import os
import sklearn.datasets as s
import dataset as d

class IrisDataset(d.DatasetI):
    def __init__(self):
        ds = s.load_iris()
        self.__data = ds.data
        self.__target = ds.target

    def name(self):
        return "iris"

    def data(self):
        return self.__data

    def target(self):
        return self.__target

class DigitsDataset(d.DatasetI):
    def __init__(self):
        ds = s.load_digits()
        self.__data = ds.data
        self.__target = ds.target

    def name(self):
        return "digits"

    def data(self):
        return self.__data

    def target(self):
        return self.__target

class Random2Dataset(d.DatasetI):
    def __init__(self):
        X, y = s.make_classification(n_samples = 5000)
        self.__data = X
        self.__target = y

    def name(self):
        return "random2"

    def data(self):
        return self.__data

    def target(self):
        return self.__target

class Random10Dataset(d.DatasetI):
    def __init__(self):
        X, y = s.make_classification( n_samples = 5000
                                    , n_classes = 10
                                    , n_informative = 7)
        self.__data = X
        self.__target = y

    def name(self):
        return "random10"

    def data(self):
        return self.__data

    def target(self):
        return self.__target

class HastieDataset(d.DatasetI):
    def __init__(self):
        X, y = s.make_hastie_10_2()
        self.__data = X
        self.__target = y

    def name(self):
        return "hastie"

    def data(self):
        return self.__data

    def target(self):
        return self.__target

class LibSVMDataset(d.DatasetI):
    def __init__(self, name):
        X, y = s.load_svmlight_file(name)
        self.__data = X.toarray()
        self.__target = y
        self.__name = name

    def name(self):
        return os.path.basename(self.__name)

    def data(self):
        return self.__data

    def target(self):
        return self.__target
