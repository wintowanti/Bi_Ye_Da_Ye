#coding=UTF-8
from jieba import cut as weibo_tokenize
from data_helper.social_text import Social_Text
class WeiBo(Social_Text):
    tokenize_f = weibo_tokenize
    def __init__(self, raw_text, raw_target, stance, flag):
        super(WeiBo, self).__init__(raw_text.decode("UTF-8"), raw_target.decode("UTF-8"), stance, flag)
        #self.tokenize_f = weibo_tokenize
