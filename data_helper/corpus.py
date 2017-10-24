# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import torch
from pandas import read_csv
import random
from collections import defaultdict, Counter, defaultdict
from data_helper.dictionary import Dictionary
from data_helper.tokenized import tokenize as twitter_tokenize
from data_helper.twitter import Seme_tweet, Multi_target_tweet

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.dictionary.add_word("_padding_")
        self.dictionary.add_word("_UNK_")
        self.tweets = []
        self.read_data(path)
        self.build_dict()
        self.idxed_data()

    def read_data(self, path):
        raise NotImplementedError()

    def idxed_data(self):
        for item in self.tweets:
            item.idxed_data_by_dict(self.dictionary)

    def build_dict(self):
        word_list = []
        for tweet in filter(lambda tweet: tweet.flag == "Train", self.tweets):
            for word in tweet.tokenize():
                word_list.append(word)
        for word, times in Counter(word_list).iteritems():
            if times >= 2:
                self.dictionary.add_word(word)
        return self.dictionary

class Multi_Target_Corpus(Corpus):
    def __init__(self, path):
        #self.check_data()
        super(Multi_Target_Corpus, self).__init__(path)
        self.pair_names = [["Donald Trump", "Hilary Clinton"],
                           ["Donald Trump","Ted Cruz"],
                           ["Hilary Clinton","Bernie Sanders"]]

    def read_data(self, path):
        data = read_csv(path, names=["tweet", "target1", "stance1", "target2", "stance2", "flag"], skiprows=[0])
        for twt, t1, s1, t2, s2, flag in zip(data.tweet, data.target1, data.stance1, data.target2, data.stance2, data.flag):
            self.tweets.append(Multi_target_tweet(twt, t1, s1, t2, s2, flag))

    def check_data(self):
        count = defaultdict(float)
        count_f_a = defaultdict(lambda :defaultdict(float))
        print("-----------------check-data start------------------")
        for mul_tweet in self.tweets:
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

    def analysis(self):
        self.analysis_tweet_len()

    def analysis_tweet_len(self):
        print("------- analysis len start-----------")
        len_list = [len(item.idx_tweet) for item in self.tweets]
        len_count = Counter(len_list)
        for len_, times in len_count.iteritems():
            print("len %d : %d"%(len_, times))
        print("------- analysis len end-----------")

    def analysis_count(self):
        word_list = []
        for tweet in filter(lambda tweet: tweet.flag == "Train", self.tweets):
            for word in tokenize(tweet.raw_tweet):
                word_list.append(word)
        tmp = Counter(word_list)
        return tmp

    def iter_epoch(self, target1, target2, flag, pair_idx, batch_size=30):
        idx_tweets = []
        stance1 = []
        stance2 = []
        for tweet in self.tweets:
            if tweet.target1 == target1 and tweet.target2 == target2 and tweet.flag == flag:
            #if tweet.flag == flag:
                idx_tweets.append(tweet.idx_tweet)
                stance1.append(tweet.stance1)
                stance2.append(tweet.stance2)

        for start_idx in range(len(idx_tweets))[::batch_size]:
            end_idx = min(start_idx + batch_size, len(idx_tweets))
            yield idx_tweets[start_idx:end_idx:], stance1[start_idx:end_idx:], stance2[start_idx:end_idx:], pair_idx

    def iter_train_epoch(self, batch_size=64, is_random=True):
        all_items = []
        for pair_idx, (target1, target2) in enumerate(self.pair_names):
            for item in self.iter_epoch(target1, target2, "Train", pair_idx, batch_size):
                all_items.append(item)
        if is_random:
            random.shuffle(all_items)
        for item in all_items:
            yield item

class Seme_Corpus(Corpus):
    def __init__(self, path):
        super(Seme_Corpus, self).__init__(path)
        self.targets = [ "Atheism",
                        "Climate Change is a Real Concern",
                        "Feminist Movement",
                        "Hillary Clinton",
                        "Legalization of Abortion"]
        #self.add_targets_word2dict()

    # def add_targets_word2dict(self):
    #     for target in self.targets:
    #         for word in twitter_tokenize(target):
    #             self.dictionary.add_word(word.lower())

    def read_semeval_data(self, path, flag):
        data = read_csv(path, names=["tweet", "target", "stance", "opinion", "sentiment"], skiprows=[0])
        for tweet, target, stance, sentiment in zip(data.tweet, data.target, data.stance, data.sentiment):
            self.tweets.append(Seme_tweet(tweet, target, stance, sentiment, flag))

    def read_data(self, path):
        self.read_semeval_data(os.path.join(path, "train.csv"), "Train")
        self.read_semeval_data(os.path.join(path, "test.csv"), "Test")

    def iter_epoch(self, target_idx, flag, batch_size=18):
        idx_tweets = []
        idx_targets = []
        stances = []
        sentiments = []
        for tweet in self.tweets:
            if tweet.target == self.targets[target_idx] and tweet.flag == flag:
                idx_tweets.append(tweet.idx_tweet)
                idx_targets.append(tweet.idx_target)
                stances.append(tweet.stance)
                sentiments.append(tweet.sentiment)

        for start_idx in range(len(idx_tweets))[::batch_size]:
            end_idx = min(start_idx + batch_size, len(idx_tweets))
            yield idx_tweets[start_idx:end_idx:], idx_targets[start_idx:end_idx:], stances[start_idx:end_idx:], sentiments[start_idx:end_idx:], target_idx

    def iter_all_train_target(self, batch_size=30, is_random=True):
        data = []
        for target_idx in range(len(self.targets)):
            for item in self.iter_epoch(target_idx, "Train", batch_size):
                data.append(item)
        if is_random is True:
            random.shuffle(data)
        for item in data:
            yield item


if __name__ == "__main__":
    #corpus = Corpus("./data/all_data_tweet_text.csv")
    semeval_corpus = Seme_Corpus("../data")
    for (idx_tweet, idx_targets, stances, sentiments, target_id) in semeval_corpus.iter_epoch(1,"Train"):
        pass
    # tsum = 0
    # for idxs, s1, s2 in corpus.iter_epoch("Donald Trump", "Hilary Clinton", "Train", batch_size=100):
    #     tsum += len(s1)
    # print(tsum)
    # corpus.analysis()
    pass