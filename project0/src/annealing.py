import re

import classification_dataset

ANNEALING_CLASS_FEATURE = "class"
ANNEALING_FEATURES = [
    "family",
    "product-type",
    "steel",
    "carbon",
    "hardness",
    "temper_rolling",
    "condition",
    "formability",
    "strength",
    "non-ageing",
    "surface-finish",
    "surface-quality",
    "enamelability",
    "bc",
    "bf",
    "bt",
    "bw",
    "bl",
    "m",
    "chrom",
    "phos",
    "cbond",
    "marvi",
    "exptl",
    "ferro",
    "corr",
    "blue",
    "lustre",
    "jurofm",
    "s",
    "p",
    "shape",
    "thick",
    "width",
    "len",
    "oil",
    "bore",
    "packing",
    "class"
    ]

KIND_ANNEALING = classification_dataset.ClassificationDatasetI()

class AnnealingInstance(classification_dataset.ClassificationInstanceI):
    def __init__(self, line):
        ziplist = zip(ANNEALING_FEATURES, re.split(r',', line))
        ziplist = filter(lambda (_,fv): fv != '?', ziplist)

        self.__features = dict(ziplist)
        self.__class = self.__features[ANNEALING_CLASS_FEATURE]
        del self.__features[ANNEALING_CLASS_FEATURE]

    def instance_class(self):
        return self.__class

    def features(self):
        return self.__features

class AnnealingDataset(classification_dataset.ClassificationDatasetI):
    def __init__(self, filename):
        with open(filename, "r") as f:
            lines = re.split(r'\n', f.read())[0:-1]
            self.__instances = [ AnnealingInstance(l) for l in lines ]

    def classes(self):
        return [ "1", "2", "3", "4", "5", "U" ]

    def instances(self):
        return self.__instances

    def name(self):
        return "annealing"

    def kind(self):
        return KIND_ANNEALING

if __name__ == '__main__':
    a = AnnealingDataset("../data/annealing/anneal.data")
    for i in a.instances():
        print i.features(), i.instance_class()
    print a.name()
    print a.classes()
