#coding=UTF-8

from pandas import read_csv
import random
from data_helper.corpus import Corpus
import os
from data_helper.weibo import WeiBo

class NLPCC_Corpus(Corpus):
    def __init__(self, path):
        super(NLPCC_Corpus, self).__init__(path)
        self.targets = [u"IphoneSE",
                        u"春节放鞭炮",
                        u"俄罗斯在叙利亚的反恐行动",
                        u"开放二胎",
                        u"深圳禁摩限电"]

    def read_nlpcc_data(self, path, flag):
        data = read_csv(path, names=["id", "target", "text", "stance"], skiprows=[0], sep="\t")
        for text, target, stance in zip(data.text, data.target, data.stance):
            self.social_texts.append(WeiBo(text, target, stance, flag))

    def read_data(self, path):
        self.read_nlpcc_data(os.path.join(path, "evasampledata4-TaskAA.txt"), "Train")
        self.read_nlpcc_data(os.path.join(path, "NLPCC_2016_Stance_Detection_Task_A_gold.txt"), "Test")
        # self.read_semeval_data(os.path.join(path, "test.csv"), "Test")

    def iter_epoch(self, target_idx, flag, batch_size=18):
        if flag == "Dev":
            flag = "Test"
        idx_text = []
        idx_targets = []
        print(self.targets[target_idx])
        stances = []
        all_len = 0
        for tweet in self.social_texts:
            if tweet.raw_target == self.targets[target_idx] and (tweet.flag == flag or flag == "Train"):
                if tweet.flag == flag:
                    all_len += 1
                idx_text.append(tweet.idx_text)
                idx_targets.append(tweet.idx_target)
                stances.append(tweet.stance)
        if flag == "Train":
            all_len += 20
        for start_idx in range(all_len)[::batch_size]:
            end_idx = min(start_idx + batch_size, all_len)
            yield idx_text[start_idx:end_idx:], idx_targets[start_idx:end_idx:], stances[start_idx:end_idx:], target_idx