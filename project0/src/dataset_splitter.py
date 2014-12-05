import copy
import math
import random

import classification_dataset as cd

class DatasetSplitterI:
    def split(self, dataset):
        """Returns a list of training and evaluation set tuples."""
        raise NotImplementedError("Please implement this yourself.")

class RatioRangeSplitter(DatasetSplitterI):
    def __init__(self, start, stop, step):
        self.__start = start
        self.__stop  = stop
        self.__step  = step

    def split(self, ds):
        instances = ds.instances()

        # Reproducible random shuffle to avoid sequential splits in test/eval.
        random.seed(0)
        random.shuffle(instances)

        for ratio in range(self.__start, self.__stop, self.__step):
            cutoff = int(math.floor(len(instances) * ratio / 100))

            trainDataset = copy.copy(ds)
            trainDataset.set_instances(instances[:cutoff])

            testDataset = copy.copy(ds)
            testDataset.set_instances(instances[cutoff:])

            yield (trainDataset, testDataset)

class RatioSplitter(DatasetSplitterI):
    def __init__(self, percent):
        self.__splitter = RatioRangeSplitter(percent, percent + 1, 1)

    def split(self, ds):
        return self.__splitter.split(ds)

class CrossfoldSplitter(DatasetSplitterI):
    def __init__(self, k):
        self.__k = k

    def split(self, ds):
        instances = ds.instances()
        cutoff = int(math.floor(len(instances) / self.__k))

        # Reproducible random shuffle to avoid sequential splits in test/eval.
        random.seed(0)
        random.shuffle(instances)

        for i in range(self.__k):
            test  = instances[:cutoff]
            train = instances[cutoff:]

            testds = copy.copy(ds)
            testds.set_instances(test)

            trainds = copy.copy(ds)
            trainds.set_instances(train)

            yield (trainds, testds)

            # Rotate instances.
            instances = train + test
