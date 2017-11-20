#coding=UTF-8

from pandas import read_csv
import random
from data_helper.corpus import Corpus
import os
from data_helper.twitter import Seme_tweet

class Seme_Corpus(Corpus):
    def __init__(self, path):
        super(Seme_Corpus, self).__init__(path)
        self.targets = [ "Atheism",
                        "Climate Change is a Real Concern",
                        "Feminist Movement",
                        "Hillary Clinton",
                        "Legalization of Abortion"]
        #self.add_targets_word2dict()

    def read_semeval_data(self, path, flag):
        data = read_csv(path, names=["tweet", "target", "stance", "opinion", "sentiment"], skiprows=[0])
        for tweet, target, stance, sentiment in zip(data.tweet, data.target, data.stance, data.sentiment):
            self.social_texts.append(Seme_tweet(tweet, target, stance, sentiment, flag))

    def read_data(self, path):
        self.read_semeval_data(os.path.join(path, "train.csv"), "Train")
        self.read_semeval_data(os.path.join(path, "test.csv"), "Test")

    def iter_epoch(self, target_idx, flag, batch_size=18, is_shuffle=True):
        print("no good")
        idx_tweets = []
        idx_targets = []
        stances = []
        sentiments = []
        for tweet in self.social_texts:
            if tweet.raw_target == self.targets[target_idx] and (tweet.flag == flag or (flag == "Dev" and tweet.flag == "Test")):
                idx_tweets.append(tweet.idx_text)
                idx_targets.append(tweet.idx_target)
                stances.append(tweet.stance)
                sentiments.append(tweet.sentiment)
        ##shuffle list
        # for item in [idx_tweets, idx_targets, stances, sentiments]:
        #     random.shuffle(item)
        if flag == "Dev":
            all_len = len(idx_tweets)
        else:
            all_len = len(idx_tweets)

        for start_idx in range(all_len)[::batch_size]:
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