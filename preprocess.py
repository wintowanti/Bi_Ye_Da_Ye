# -*- coding: utf-8 -*-
import numpy as np
import os

def init_embedding(word_dict, embed_size, embedding_path, is_random=False):
    voc_size = len(word_dict)
    #embedding_matrix = np.random.normal(loc=0.0, scale=1.0, size=(voc_size, embed_size))
    embedding_matrix = np.random.uniform(low=-1, high=1, size=(voc_size, embed_size))

    bingo_count = 0
    if is_random == False:
        assert os.path.exists(embedding_path) == True
        with open(embedding_path) as f:
            for line in f.readlines():
                line = line.strip()
                target_word = line.split(" ")[0]
                if target_word in word_dict:
                    print("bingo word %s"%(target_word))
                    bingo_count += 1
                    idx = word_dict[target_word]
                    embedding_matrix[idx] = np.array([float(num) for num in line.split(" ")[1::]])
    print "bingo count %d"%bingo_count
    return embedding_matrix


