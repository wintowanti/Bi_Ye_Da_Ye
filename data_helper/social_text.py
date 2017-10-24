#coding=UTF-8
class Social_Text(object):
    def __init__(self, raw_text, raw_target, stance, flag):
        self.raw_text = raw_text
        self.raw_target = raw_target
        self.stance = stance
        self.flag = flag
        self.tokenized_text = [word for word in self.tokenize()]
        self.tokenized_target =[word for word in self.tokenize_target()]

    def idxed_data_by_dict(self, dict):
        self.idx_text = []
        self.idx_target = []
        #idxed tweet
        for word in self.tokenize():
            if word not in dict.word2idx:
                word = "_UNK_"
            self.idx_text.append(dict.word2idx[word])

        #idxed target
        for word in self.tokenize_target():
            if word not in dict.word2idx:
                dict.add_word(word)
            self.idx_target.append(dict.word2idx[word])

    def tokenize(self):
        #lower the raw twitter to tokenize
        return self.tokenize_f(self.raw_text.lower())

    def tokenize_target(self):
        #lower the raw twitter to tokenize
        return self.tokenize_f(self.raw_target.lower())