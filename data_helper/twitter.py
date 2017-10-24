#coding=UTF-8
from data_helper.tokenized import tokenize as twiiter_tokenize

class Multi_target_tweet(object):
    def __init__(self, raw_tweet, target1, stance1, target2, stance2, flag):
        self.raw_tweet = raw_tweet
        self.target1, self.stance1 = target1, stance1
        self.target2, self.stance2 = target2, stance2

        self.flag = flag

    def tokenize_by_dict(self, dict):
        self.idx_tweet = []
        for word in self.tokenize():
            if word not in dict.word2idx:
                word = "_UNK_"
            self.idx_tweet.append(dict.word2idx[word])

    def tokenize(self):
        #lower the raw twitter to tokenize
        return twiiter_tokenize(self.raw_tweet.lower())


class Seme_tweet(object):
    def __init__(self, raw_tweet, target, stance, sentiment, flag):
        self.raw_tweet = raw_tweet
        self.target, self.stance = target, stance
        self.flag = flag
        self.sentiment = sentiment

    def idxed_data_by_dict(self, dict):
        self.idx_tweet = []
        self.idx_target = []
        #idxed tweet
        for word in self.tokenize():
            if word not in dict.word2idx:
                word = "_UNK_"
            self.idx_tweet.append(dict.word2idx[word])

        #idxed target
        for word in self.tokenize_target():
            if word not in dict.word2idx:
                dict.add_word(word)
            self.idx_target.append(dict.word2idx[word])

    def tokenize(self):
        #lower the raw twitter to tokenize
        return twiiter_tokenize(self.raw_tweet.lower())

    def tokenize_target(self):
        #lower the raw twitter to tokenize
        return twiiter_tokenize(self.target.lower())
