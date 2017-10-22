#coding=UTF-8
from __future__ import print_function

class Config(object):
    data_path = "./data/all_data_tweet_text.csv"
    seme_data_path = "./data"
    embedding_path = "./data/glove.twitter.27B.100d.txt"
    embedding_size = 100
    fixed_len = 30
    epoch = 150
    batch_size= 64

    voc_len = None
    embedding_matrix = None

    #lstm
    hidden_size = 64
    class_size = 3


if __name__ == "__main__":
    print("ok")