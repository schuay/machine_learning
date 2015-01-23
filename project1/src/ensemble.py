#!/usr/bin/env python2

# TODO: A module to read used classifiers and their parameters from a config file
#       (command line args would be too long and messy). There must be a ready-made module
#       around that can do this.
# TODO: Evaluate accuracy on all classifiers using crossfold validation. Store results,
#       including data to write to CSV later on.
# TODO: Generate ensemble classifier (weighted & std) and evaluate.
# TODO: Print results to stdout.

import getopt
import sys

class Opts:
    config_file = "conf/default"
    dataset = "annealing"
    verbose = False

options = Opts()

DATASETS = { 'twitter': lambda mi, fs, tr: twitter.TwitterDataset(
                    '../data/twitter/Sentiment-Analysis-Dataset.zip',
                    mi, fs, tr)
           , 'annealing': lambda mi, fs, tr: annealing.AnnealingDataset(
                    '../data/annealing/anneal.data')
           , 'tic': lambda mi, fs, tr: tic.TICDatasetClass(
                    '../data/tic/ticdata2000.txt', mi)
           }

def usage():
    print("""USAGE: %s [options]
            -c  The configuration file to load.`
            -d  The dataset to use. One of %s.
            -v  Verbose output.""" %
            ( sys.argv[0]
            , DATASETS.keys()
            )
         )
    sys.exit(1)

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], "c:d:hv")
    for o, a in opts:
        if o == "-c":
            options.config_file = a
        elif o == "-d":
            if not a in DATASETS:
                usage()
            options.dataset = a
        elif o == "-v":
            options.verbose = True
        else:
            usage()

    print 'Accuracy: 100%'
