import copy
import math

import classification_dataset as cd

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

        trainDataset = copy.copy(ds)
        trainDataset.set_instances(instances[:cutoff])

        testDataset = copy.copy(ds)
        testDataset.set_instances(instances[cutoff:])

        return [(trainDataset, testDataset)]
