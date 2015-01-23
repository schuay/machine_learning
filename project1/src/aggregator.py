#!/usr/bin/env python2

import tweet
import pickle

class AggregatorI:
    def add(t, sentiment):
        raise NotImplementedError("Please implement this yourself.")

    def get_sentiment(self):
        raise NotImplementedError("Please implement this yourself.")

    def get_last_id(self):
        raise NotImplementedError("Please implement this yourself.")

    def get_num(self):
        raise NotImplementedError("Please implement this yourself.")

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

class MeanAggregator(AggregatorI):
    def __init__(self):
        self.__last_id = 0
        self.__num = 0.0
        self.__sentiment_ctr = 0.0

    def add(self, t, sentiment):
        self.__sentiment_ctr += sentiment
        self.__num += 1.0
        self.__last_id = t[tweet.ID]

    def get_sentiment(self):
        if (self.__num == 0.0):
            return None
        return self.__sentiment_ctr / self.__num

    def get_last_id(self):
        return self.__last_id

    def get_num(self):
        return self.__num

class RetweetWeightedAggregator(AggregatorI):
    def __init__(self):
        self.__last_id = 0
        self.__num = 0
        self.__weighted_num = 0.0
        self.__sentiment_ctr = 0.0

    def add(self, t, sentiment):
        weight = t[tweet.RETWEET_COUNT] + 1

        self.__sentiment_ctr += weight * sentiment
        self.__weighted_num += weight
        self.__num += 1
        self.__last_id = t[tweet.ID]

    def get_sentiment(self):
        if (self.__weighted_num == 0.0):
            return None
        return self.__sentiment_ctr / self.__weighted_num

    def get_last_id(self):
        return self.__last_id

    def get_num(self):
        return self.__num
