import regression_dataset as rd

KIND_SOLAR_FLARES = rd.RegressionDatasetI()

class SolarFlaresInstance(rd.RegressionInstanceI):
    def __init__(self, row):
        self.__features = row

    def features(self):
        return self.__features

    def __str__(self):
        return "%s" % self.features()

class SolarFlaresDataset(rd.RegressionDatasetI):
    def __init__(self, data_archive):
        self.__instances = []

        with open(data_archive, 'r') as f:
            next(f)
            for row in f:
                self.__instances.append(
                        SolarFlaresInstance(row.strip().split(" ")))

    def instances(self):
        return self.__instances

    def name(self):
        return "solar_flares"

    def kind(self):
        return KIND_SOLAR_FLARES

import sys

if __name__ == '__main__':
    dataset = SolarFlaresDataset(sys.argv[1])
    for instance in dataset.instances():
        print(instance)
