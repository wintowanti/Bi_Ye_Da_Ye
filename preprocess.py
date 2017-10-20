# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import os

def init_embedding(word_dict, embed_size, embedding_path, is_random=False):
    voc_size = len(word_dict)
    #embedding_matrix = np.random.normal(loc=0.0, scale=1.0, size=(voc_size, embed_size))
    #np.random.seed(13)
    embedding_matrix = np.random.uniform(low=-1, high=1, size=(voc_size, embed_size))

    bingo_count = 0
    #embedding_matrix[0] = np.array([0.0 for _ in range(embed_size)])
    if is_random == False:
        assert os.path.exists(embedding_path) == True
        with open(embedding_path) as f:
            for line in f.readlines():
                line = line.strip()
                target_word = line.split(" ")[0]
                if target_word in word_dict:
                    #print("bingo word %s"%(target_word))
                    bingo_count += 1
                    idx = word_dict[target_word]
                    embedding_matrix[idx] = np.array([float(num) for num in line.split(" ")[1::]])
    print "bingo count %d"%bingo_count
    return embedding_matrix


def fixed_length(inputs, fixed_len):
    batch_len = len(inputs)
    outputs = np.zeros(shape=(batch_len, fixed_len), dtype=np.int64)
    for idx, vals in enumerate(inputs):
        end = min(fixed_len, len(vals))
        outputs[idx, 0:end] = vals[0:end]
    return outputs


def stance2idx(stances):
    idxs = [hash_stance(stance) for stance in stances]
    return np.array(idxs)


def hash_stance(stance):
    if stance == "FAVOR" or stance == "pos":
        return 0
    if stance == "NONE" or stance == "other":
        return 1
    if stance == "AGAINST" or stance == "neg":
        return 2
    raise Exception("can't hash stance: "+stance)


def metric(y_pred, y_true, name):

    precision, recall, f1_score, support = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)
    #print confusion_matrix(y_true=y_true, y_pred=y_pred)
    f1_average = (f1_score[0] + f1_score[2]) / 2

    print("n: %s Favor f1: %f  Against f1 %f average: %f"%(
        name, f1_score[0], f1_score[2], f1_average))
    return f1_average




