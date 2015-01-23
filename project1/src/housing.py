import regression_dataset as rd
import csv
import datetime

KIND_Housing = rd.RegressionDatasetI()

class HousingInstance(rd.RegressionInstanceI):
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

class HousingDataset(rd.RegressionDatasetI):
    def __init__(self, data_file, max_instances):
        self.__instances = []

        instance_count = 0
        with open(data_file, 'r') as f:
            for row in f:
                self.__instances.append(HousingInstance(row.split()))
                instance_count += 1
                if max_instances is not None and instance_count >= max_instances:
                    break

    def instances(self):
        return self.__instances

    def set_instances(self, instances):
        self.__instances = instances

    def name(self):
        return "housing"

    def kind(self):
        return KIND_Housing

import sys

if __name__ == '__main__':
    dataset = HousingDataset(sys.argv[1], 1000)
    for instance in dataset.instances():
        print(instance)
