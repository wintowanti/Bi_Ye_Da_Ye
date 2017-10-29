#coding=UTF-8
from __future__ import print_function

class Config(object):
    data_path = "./data/all_data_tweet_text.csv"
    seme_data_path = "./data"
    nlpcc_data_path = "./data/NLPCC"
    embedding_path = "./data/glove.twitter.27B.100d.txt"
    weibo_embedding_path = "./data/weibo_w2v_200.txt"
    embedding_size = 200
    weibo_embedding_size = 200
    fixed_len = 50
    epoch = 150
    batch_size= 64

    voc_len = None
    embedding_matrix = None

    #lstm
    hidden_size = 64
    class_size = 3


if __name__ == "__main__":
    print("ok")