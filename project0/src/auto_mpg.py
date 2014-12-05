import regression_dataset as rd
import csv
import datetime

KIND_AutoMPG = rd.RegressionDatasetI()

class AutoMPGInstance(rd.RegressionInstanceI):
    def __init__(self, row):
        row = [float('nan') if (c == '?' or c == '') else float(c) for c in row[:8]]

        float_features = [ float(col) for col in row[:7] ]
        origin_features = [ 1.0 if int(row[7]) == o else 0 for o in range(1, 4) ]

        self.__features = float_features + origin_features

    def features(self):
        return self.__features

    def x(self):
        return self.features()[1:]

    def y(self):
        return [self.features()[0]]

    def __str__(self):
        return "%s" % self.features()

class AutoMPGDataset(rd.RegressionDatasetI):
    def __init__(self, data_file, max_instances):
        self.__instances = []

        instance_count = 0
        with open(data_file, 'r') as f:
            for row in f:
                self.__instances.append(AutoMPGInstance(row.split()))
                instance_count += 1
                if max_instances is not None and instance_count >= max_instances:
                    break

    def instances(self):
        return self.__instances

    def set_instances(self, instances):
        self.__instances = instances

    def name(self):
        return "auto_mpg"

    def kind(self):
        return KIND_AutoMPG

import sys

if __name__ == '__main__':
    dataset = AutoMPGDataset(sys.argv[1], 1000)
    for instance in dataset.instances():
        print(instance)
