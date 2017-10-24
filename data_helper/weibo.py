#coding=UTF-8

class WeiBo(object):
    def __int__(self, raw_text, raw_target, stance):
        self.raw_text = raw_text
        self.raw_target  = raw_target
        self.stance = stance

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