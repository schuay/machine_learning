import classification_dataset as cd
import tweet
import zipfile

COMPRESSED_FILENAME = "Sentiment Analysis Dataset.csv"

IX_CLASS = 1
IX_TEXT  = 3

# TODO: Possibly take a look at further filtering and preprocessing: quotes,
# punctuation, etc.

class TwitterInstance(cd.ClassificationInstanceI):
    def __init__(self, row, feature_selection, transformer):
        split_row = row.split(",", 3)

        self.__instance_class = int(split_row[IX_CLASS])
        self.__text = split_row[IX_TEXT]
        self.__features = feature_selection.select_features(
                transformer.transform({tweet.TEXT: self.__text}))

    def instance_class(self):
        return self.__instance_class

def features(self):
        return self.__features

    def __str__(self):
        return "%d: %s" % (self.instance_class(), self.features())

class TwitterDataset(cd.ClassificationDatasetI):
    NEG = 0
    POS = 1

    def __init__(self, data_archive, feature_selection, transformer):
        self.__instances = []

        with zipfile.ZipFile(data_archive, 'r') as f:
            datafile = f.open(COMPRESSED_FILENAME, "rU")
            next(datafile)  # Skip the header.
            for row in datafile:
                self.__instances.append(
                    TwitterInstance(row, feature_selection, transformer))

    def classes(self):
        return [POS, NEG]

    def instances(self):
        return self.__instances

    def name(self):
        return "twitter"

import featureselection as fs
import sys
import transformer as tr

if __name__ == '__main__':
    feature_selection = fs.AllFeatures(
                                [ fs.NGram(fs.StopWordFilter(fs.AllWords()), 1)
                                , fs.Emoticons()
                                ])
    transformer = tr.IdentityTransformer()

    dataset = TwitterDataset(sys.argv[1], feature_selection, transformer)
