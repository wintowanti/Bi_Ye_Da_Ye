# -*- coding: utf-8 -*-

from data_helper import Corpus
from preprocess import init_embedding

data_path = "./data/all_data_tweet_text.csv"
embedding_path = "./data/glove.twitter.27B.100d.txt"

def main():
    corpus = Corpus(data_path)
    embedding_matrix = init_embedding(corpus.dictionary.word2idx, 100, embedding_path)
    pass

if __name__ == "__main__":
    main()
