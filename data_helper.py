# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import torch
from pandas import read_csv
from collections import defaultdict, Counter
from tokenized import tokenize

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Multi_target_tweet(object):
    def __init__(self, raw_tweet, target1, stance1, target2, stance2, flag):
        self.raw_tweet = raw_tweet

        self.target1, self.stance1 = target1, stance1
        self.target2, self.stance2 = target2, stance2

        self.flag = flag

    def tokenize_(self, dict):
        self.idx_tweet = []
        for word in tokenize(self.raw_tweet):
            dict.add_word(word)
            self.idx_tweet.append(dict.word2idx[word])

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.multi_target_tweets = []
        self.read_data(path)
        #self.check_data()
        self.tokenize()
        pass


    def read_data(self, path):
        data = read_csv(path, names=["tweet", "target1", "stance1", "target2", "stance2", "flag"], skiprows=[0])
        for twt, t1, s1, t2, s2, flag in zip(data.tweet, data.target1, data.stance1, data.target2, data.stance2, data.flag):
            self.multi_target_tweets.append(Multi_target_tweet(twt, t1, s1, t2, s2, flag))
        pass

    def check_data(self):
        count = defaultdict(float)
        count_f_a = defaultdict(lambda :defaultdict(float))
        print("-----------------check-data start------------------")
        for mul_tweet in self.multi_target_tweets:
            key = "-".join([mul_tweet.target1, mul_tweet.target2, mul_tweet.flag])
            key1 = "-".join([mul_tweet.target1, mul_tweet.target2])
            key2 = "-".join([mul_tweet.stance1, mul_tweet.stance2])
            count_f_a[key1][key2] += 1.0
            count[key] += 1.0
        for key, v in count.iteritems():
            print("%s : %d"%(key, v))

        for key1, items in count_f_a.iteritems():
            print(" key1: %s"%key1)
            tsum = sum(items.values())
            for key2, val in items.iteritems():
                print("key2 %s: %.1f "%(key2, val*100/tsum))

        print("-----------------check-data end------------------")

    def tokenize(self):
        for item in self.multi_target_tweets:
            item.tokenize_(self.dictionary)

    def analysis(self):
        self.analysis_tweet_len()

    def analysis_tweet_len(self):
        print("------- analysis len start-----------")
        len_list = [len(item.idx_tweet) for item in self.multi_target_tweets]
        len_count = Counter(len_list)
        for len_, times in len_count.iteritems():
            print("len %d : %d"%(len_, times))
        print("------- analysis len end-----------")


if __name__ == "__main__":
    corpus = Corpus("./data/all_data_tweet_text.csv")
    corpus.analysis()
    pass
