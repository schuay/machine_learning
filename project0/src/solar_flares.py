import regression_dataset as rd


COLUMNS = [ "class"
          , "size"
          , "distribution"
          , "activity"
          , "evolution"
          , "prev_activity"
          , "hist_complex"
          , "region_hist_complex"
          , "area"
          , "max_spot_area"
          , "c_class_flares"
          , "m_class_flares"
          , "x_class_flares"
          ]

KIND_SOLAR_FLARES = rd.RegressionDatasetI()

def _split_attr(attr, value):
    return 1 if attr == value else 0

# TODO: Make this generic, add variations, etc. See also
# tweet transformers / feature selectors.
class SolarFlaresPreprocessor:
    @staticmethod
    def x(features):
        return features[:24]

    @staticmethod
    def y(features):
        return features[24:]

    @staticmethod
    def process(features):
        return [ _split_attr(features[0], "A")
               , _split_attr(features[0], "B")
               , _split_attr(features[0], "C")
               , _split_attr(features[0], "D")
               , _split_attr(features[0], "E")
               , _split_attr(features[0], "F")
               , _split_attr(features[0], "H")
               , _split_attr(features[1], "X")
               , _split_attr(features[1], "R")
               , _split_attr(features[1], "S")
               , _split_attr(features[1], "A")
               , _split_attr(features[1], "H")
               , _split_attr(features[1], "K")
               , _split_attr(features[2], "X")
               , _split_attr(features[2], "O")
               , _split_attr(features[2], "I")
               , _split_attr(features[2], "C")
               , int(features[3])
               , int(features[4])
               , int(features[5])
               , int(features[6])
               , int(features[7])
               , int(features[8])
               , int(features[9])
               , int(features[10])
               , int(features[11])
               , int(features[12])
               ]

class SolarFlaresInstance(rd.RegressionInstanceI):
    def __init__(self, row):
        self.__preprocessor = SolarFlaresPreprocessor()
        self.__features = self.__preprocessor.process(row)

    def features(self):
        return self.__features

    def x(self):
        return self.__preprocessor.x(self.__features)

    def y(self):
        return self.__preprocessor.y(self.__features)

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

    def set_instances(self, instances):
        self.__instances = instances

    def name(self):
        return "solar_flares"

    def kind(self):
        return KIND_SOLAR_FLARES

import sys

if __name__ == '__main__':
    dataset = SolarFlaresDataset(sys.argv[1])
    for instance in dataset.instances():
        print(instance.x())
        print(instance.y())
