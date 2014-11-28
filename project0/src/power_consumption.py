import regression_dataset as rd
import csv
import zipfile
import datetime

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

class IdentityTransformer:
    def transform(self, obj):
        return obj

class ZScoreTransformer:
    def __init__(self, mu, sigma):
        self.__mu = mu
        self.__sigma = sigma

    def transform(self, obj):
        return (obj - self.__mu) / self.__sigma

def calc_mu(xs):
    return sum(xs)/len(xs)

def calc_sigma(xs):
    mu = calc_mu(xs)
    return sum([(x - mu)**2 for x in xs])/len(xs)

class PowerConsumptionInstance(rd.RegressionInstanceI):
    def __init__(self, row, hourTrans, minTrans):
        self.__features = list()

        day, month, year = row[0].split("/")
        date = datetime.date(int(year), int(month), int(day))
        for i in range(0,7):
            self.__features.append(1 if (date.weekday() == i) else 0)

        hours, minutes, seconds = row[1].split(":")
        self.__features.append(hourTrans.transform(float(hours)))
        self.__features.append(minTrans.transform(float(minutes)))

        # TODO: use the mean for missing values
        self.__features += [0.0 if (c == '?' or c == '') else float(c) for c in row[2:]]

    def features(self):
        return self.__features

    def x(self):
        return self.features()[:9]

    def y(self):
        return self.features()[9:]

    def __str__(self):
        return "%s" % self.features()

class PowerConsumptionDataset(rd.RegressionDatasetI):
    def __init__(self, data_archive, max_instances, zScoreTime=True):
        self.__instances = []

        instance_count = 0
        rows = list()
        with zipfile.ZipFile(data_archive, 'r') as f:
            datafile = f.open(COMPRESSED_FILENAME, "rU")
            datareader = csv.reader(datafile, delimiter = ";")
            next(datareader)
            for row in datareader:
                rows.append(row)
                instance_count += 1
                if max_instances is not None and instance_count >= max_instances:
                    break

        if zScoreTime:
            hMu = calc_mu([float(row[1].split(":")[0]) for row in rows])
            hSigma = calc_sigma([float(row[1].split(":")[0]) for row in rows])
            mMu = calc_mu([float(row[1].split(":")[1]) for row in rows])
            mSigma = calc_sigma([float(row[1].split(":")[1]) for row in rows])

        for row in rows:
            if zScoreTime:
                hourTrans = ZScoreTransformer(hMu, hSigma)
                minTrans = ZScoreTransformer(mMu, mSigma)
                inst = PowerConsumptionInstance(row, hourTrans, minTrans)
            else:
                inst = PowerConsumptionInstance(row, IdentityTransformer(),
                        IdentityTransformer())

            self.__instances.append(inst)

    def instances(self):
        return self.__instances

    def set_instances(self, instances):
        self.__instances = instances

    def name(self):
        return "power_consumption"

    def kind(self):
        return KIND_POWER_CONSUMPTION

import sys

if __name__ == '__main__':
    dataset = PowerConsumptionDataset(sys.argv[1], 1000)
    for instance in dataset.instances():
        print(instance)
