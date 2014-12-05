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

COLUMNS = [
          "MOSTYPE",
          "MAANTHUI",
          "MGEMOMV",
          "MGEMLEEF",
          "MOSHOOFD",
          "MGODRK",
          "MGODPR",
          "MGODOV",
          "MGODGE",
          "MRELGE",
          "MRELSA",
          "MRELOV",
          "MFALLEEN",
          "MFGEKIND",
          "MFWEKIND",
          "MOPLHOOG",
          "MOPLMIDD",
          "MOPLLAAG",
          "MBERHOOG",
          "MBERZELF",
          "MBERBOER",
          "MBERMIDD",
          "MBERARBG",
          "MBERARBO",
          "MSKA",
          "MSKB",
          "MSKB",
          "MSKC",
          "MSKD",
          "MHHUUR",
          "MHKOOP",
          "MAUT",
          "MAUT",
          "MAUT",
          "MZFONDS",
          "MZPART",
          "MINKM",
          "MINK",
          "MINK",
          "MINK",
          "MINK",
          "MINKGEM",
          "MKOOPKLA",
          "PWAPART",
          "PWABEDR",
          "PWALAND",
          "PPERSAUT",
          "PBESAUT",
          "PMOTSCO",
          "PVRAAUT",
          "PAANHANG",
          "PTRACTOR",
          "PWERKT",
          "PBROM",
          "PLEVEN",
          "PPERSONG",
          "PGEZONG",
          "PWAOREG",
          "PBRAND",
          "PZEILPL",
          "PPLEZIER",
          "PFIETS",
          "PINBOED",
          "PBYSTAND",
          "AWAPART",
          "AWABEDR",
          "AWALAND",
          "APERSAUT",
          "ABESAUT",
          "AMOTSCO",
          "AVRAAUT",
          "AAANHANG",
          "ATRACTOR",
          "AWERKT",
          "ABROM",
          "ALEVEN",
          "APERSONG",
          "AGEZONG",
          "AWAOREG",
          "ABRAND",
          "AZEILPL",
          "APLEZIER",
          "AFIETS",
          "AINBOED",
          "ABYSTAND",
          ]

class TICInstanceClass(TICInstance):
    def instance_class(self):
        return TICInstance.features(self)[-1]

    def features(self):
        return dict(zip(COLUMNS, TICInstance.features(self)[:-1]))

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

class TICDatasetClass(TICDataset):
    def __init__(self, data_file, max_instances):
        self.__instances = []

        instance_count = 0
        with open(data_file, 'r') as f:
            datareader = csv.reader(f, delimiter = "\t")
            for row in datareader:
                self.__instances.append(TICInstanceClass(row))
                instance_count += 1
                if max_instances is not None and instance_count >= max_instances:
                    break

    def instances(self):
        return self.__instances

    def set_instances(self, instances):
        self.__instances = instances

    def classes(self):
        keys = {}
        for e in self.__instances:
            keys[e.instance_class()] = 1
        return keys.keys()

import sys

if __name__ == '__main__':
    dataset = TICDataset(sys.argv[1], 1000)
    for instance in dataset.instances():
        print(instance)
