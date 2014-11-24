import regression_dataset as rd
import csv
import zipfile

COMPRESSED_FILENAME = "household_power_consumption.txt"

COLUMNS = [ "Date"
          , "Time"
          , "Global_active_power"
          , "Global_reactive_power"
          , "Voltage"
          , "Global_intensity"
          , "Sub_metering_1"
          , "Sub_metering_2"
          , "Sub_metering_3"
          ] 

KIND_POWER_CONSUMPTION = rd.RegressionDatasetI()

class PowerConsumptionInstance(rd.RegressionInstanceI):
    def __init__(self, row):
        self.__features = row

    def features(self):
        return self.__features

    def __str__(self):
        return "%s" % self.features()

class PowerConsumptionDataset(rd.RegressionDatasetI):
    def __init__(self, data_archive, max_instances):
        self.__instances = []

        instance_count = 0
        with zipfile.ZipFile(data_archive, 'r') as f:
            datafile = f.open(COMPRESSED_FILENAME, "rU")
            datareader = csv.reader(datafile, delimiter = ";")
            next(datareader)
            for row in datareader:
                self.__instances.append(PowerConsumptionInstance(row))
                instance_count += 1
                if max_instances is not None and instance_count >= max_instances:
                    break

    def instances(self):
        return self.__instances

    def name(self):
        return "power_consumption"

    def kind(self):
        return KIND_POWER_CONSUMPTION

import sys

if __name__ == '__main__':
    dataset = PowerConsumptionDataset(sys.argv[1], 1000)
    for instance in dataset.instances():
        print(instance)
