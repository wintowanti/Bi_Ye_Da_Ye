#coding=UTF-8
from data_helper.tokenized import tokenize as twitter_tokenize
from data_helper.social_text import Social_Text

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
        return twitter_tokenize(self.raw_tweet.lower())


class Seme_tweet(Social_Text):
    tokenize_f = twitter_tokenize
    def __init__(self, raw_tweet, target, stance, sentiment, flag):
        super(Seme_tweet, self).__init__(raw_tweet, target, stance, flag)
        self.sentiment = sentiment
        #self.tokenize_f = twiiter_tokenize
