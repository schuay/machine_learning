import regression_dataset as rd
import csv
import datetime

KIND_TIC = rd.RegressionDatasetI()

class TICInstance(rd.RegressionInstanceI):
    def __init__(self, row):
        self.__features = [ float(col) for col in row ]

    def features(self):
        return self.__features

    def x(self):
        return self.features()[:-1]

    def y(self):
        return [self.features()[-1]]

    def __str__(self):
        return "%s" % self.features()

class TICDataset(rd.RegressionDatasetI):
    def __init__(self, data_file, max_instances):
        self.__instances = []

        instance_count = 0
        with open(data_file, 'r') as f:
            datareader = csv.reader(f, delimiter = "\t")
            for row in datareader:
                self.__instances.append(TICInstance(row))
                instance_count += 1
                if max_instances is not None and instance_count >= max_instances:
                    break

    def instances(self):
        return self.__instances

    def set_instances(self, instances):
        self.__instances = instances

    def name(self):
        return "tic"

    def kind(self):
        return KIND_TIC

import sys

if __name__ == '__main__':
    dataset = TICDataset(sys.argv[1], 1000)
    for instance in dataset.instances():
        print(instance)
