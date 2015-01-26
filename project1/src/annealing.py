import numpy as np
import re

from sklearn.preprocessing import LabelEncoder

import dataset

class AnnealingDataset(dataset.DatasetI):
    def __init__(self, filename):
        with open(filename, "r") as f:
            lines = re.split(r'\n', f.read())[0:-1]

        data = []
        target = []

        for line in lines:
            fields = re.split(r',', line)
            data.append(fields[:-1])
            target.append(fields[-1])

        npdata = np.array(data)
        nptarget = np.array(target)

        # Scikit-learn requires numeric values.

        le = LabelEncoder()
        le.fit(nptarget)
        self.__target = le.transform(nptarget)

        nrows, ncols = npdata.shape
        self.__data = np.zeros((nrows, ncols), dtype = np.int64)
        for ix in xrange(ncols):
            col = npdata[:, ix]
            le.fit(col)
            self.__data[:, ix] = le.transform(col)

    def data(self):
        return self.__data

    def target(self):
        return self.__target

    def name(self):
        return "annealing"

if __name__ == '__main__':
    a = AnnealingDataset("../data/annealing/anneal.data")
    print a.name()
    print a.data()
    print a.target()
