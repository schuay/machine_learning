import math

import classification_dataset as cd

class GenericDataset(cd.ClassificationDatasetI):
    def __init__(self, classes, instances, name, kind):
        self.__classes = classes
        self.__instances = instances
        self.__name = name
        self.__kind = kind

    def classes(self):
        return self.__classes

    def instances(self):
        return self.__instances

    def name(self):
        return self.__name

    def kind(self):
        return self.__kind

class DatasetSplitterI:
    def split(self, dataset):
        """Returns a list of training and evaluation set tuples."""
        raise NotImplementedError("Please implement this yourself.")

class RatioSplitter(DatasetSplitterI):
    def __init__(self, ratio):
        self.__ratio = ratio

    def split(self, ds):
        instances = ds.instances()
        cutoff = int(math.floor(len(instances) * self.__ratio))

        trainDataset = GenericDataset(ds.classes(), instances[:cutoff],
                                      ds.name() + "_train", ds.kind())
        testDataset = GenericDataset(ds.classes(), instances[cutoff:],
                                     ds.name() + "_test", ds.kind())

        return [(trainDataset, testDataset)]
