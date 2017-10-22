# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

from data_helper import Corpus,Seme_Corpus
from preprocess import init_embedding, fixed_length, stance2idx, metric
from config import Config
from model import Text_Condition_Encoder, Fast_Text, Text_CNN, Text_LSTM, Text_GRU, Bi_GRU_CNN, Bi_GRU, Attention_Bi_GRU
from model import Attention_Bi_GRU_CNN

import torch
from torch.autograd import Variable
from torch.nn import NLLLoss,CrossEntropyLoss
import numpy as np

def train(model, config, loss_fun, optim, target_idx):
    model.train()
    corpus = config.corpus
    all_loss = 0.0
    sample_len = 0
    for id_, (idx_tweets, idx_target, stances, sentiments, target_idx) in enumerate(corpus.iter_epoch(target_idx, "Train", batch_size=16)):
    #for id_, (idx_tweets, idx_target, stances, sentiments, target_idx) in enumerate(corpus.iter_all_train_target(batch_size=30, is_random=False)):
        np_idx_tweets = fixed_length(idx_tweets, Config.fixed_len)
        np_idx_targets = fixed_length(idx_target, len(idx_target[0]))
        sample_len += len(idx_tweets)

        var_s1 = Variable(torch.from_numpy(stance2idx(stances)))
        var_s2 = Variable(torch.from_numpy(stance2idx(sentiments)))
        output1, output2 = model(np_idx_tweets, np_idx_targets, target_idx)

        #loss = loss_fun(output1, var_s1) + loss_fun(output2, var_s2)
        #loss = loss_fun(output1, var_s1)
        #loss = loss_fun(output2, var_s2)
        #train - independed
        loss = loss_fun(output1, var_s1)
        all_loss += loss.data.sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
    print("-----------train mean loss: ", all_loss/sample_len)

def test(model, config, loss_fun, flag, target_idx):
    model.eval()
    corpus = config.corpus
    all_loss = 0.0
    sample_len = 0
    y_true_stances = np.array([])
    y_pred_stances = np.array([])

    #y_true_sentiments = np.array([])
    #y_pred_sentiments = np.array([])

    for id_, (idx_tweets,idx_target, stances, sentiments, target_idx) in enumerate(corpus.iter_epoch(target_idx, flag)):
        np_idx_tweets = fixed_length(idx_tweets, Config.fixed_len)
        np_idx_targets = fixed_length(idx_target, len(idx_target[0]))
        sample_len += len(stances)

        y_true_stances = np.append(y_true_stances, stance2idx(stances))
        #y_true_sentiments = np.append(y_true_sentiments, stance2idx(sentiments))

        var_s1 = Variable(torch.from_numpy(stance2idx(stances)))
        #var_s2 = Variable(torch.from_numpy(stance2idx(sentiments)))
        output1, output2 = model(np_idx_tweets, np_idx_targets, target_idx)

        y_pred_stances = np.append(y_pred_stances, torch.max(output1, dim=1, keepdim=False)[1].data.numpy())
        #y_pred_sentiments = np.append(y_pred_sentiments, torch.max(output2, dim=1, keepdim=False)[1].data.numpy())
        loss = loss_fun(output1, var_s1)
        all_loss += loss.data.sum()
    f1_target1 = metric(y_true=y_true_stances, y_pred=y_pred_stances, name="target1")
    #f1_target2 = metric(y_true=y_true_sentiments, y_pred=y_pred_sentiments, name="target2")
    print(flag+" mean loss: ", all_loss/sample_len)
    print("f1: %f"%f1_target1)
    return f1_target1, y_true_stances, y_pred_stances

def test_all_target(model, Config, loss_fun):
    f1_targets = []
    y_true_stances = np.array([])
    y_pred_stances = np.array([])
    for target_idx in range(len(Config.corpus.targets)):
        print("target: %s"%(Config.corpus.targets[target_idx]))
        f1, t, p = test(model, Config, loss_fun, "Train", target_idx)
        f1, t, p = test(model, Config, loss_fun, "Test", target_idx)
        f1_targets.append(f1)
        y_true_stances = np.append(y_true_stances, t)
        y_pred_stances = np.append(y_pred_stances, p)
    micro_f1 = metric(y_true_stances, y_pred_stances, "micro-F1")
    return micro_f1

def test_target(model, Config, loss_fun, target_idx):
    f1_targets = []
    y_true_stances = np.array([])
    y_pred_stances = np.array([])
    print("target: %s"%(Config.corpus.targets[target_idx]))
    f1, t, p = test(model, Config, loss_fun, "Train", target_idx)
    f1, t, p = test(model, Config, loss_fun, "Test", target_idx)
    f1_targets.append(f1)
    y_true_stances = np.append(y_true_stances, t)
    y_pred_stances = np.append(y_pred_stances, p)
    micro_f1 = metric(y_true_stances, y_pred_stances, "micro-F1")
    return micro_f1

def main():
    corpus = Seme_Corpus(Config.seme_data_path)
    embedding_matrix = init_embedding(corpus.dictionary.word2idx, Config.embedding_size, Config.embedding_path)
    Config.voc_len = len(corpus.dictionary)
    Config.embedding_matrix = embedding_matrix
    model = Attention_Bi_GRU_CNN(Config)
    #model = Attention_Bi_GRU(Config)
    #model = Bi_GRU_CNN(Config)
    #model = Text_GRU(Config)
    #model = Bi_GRU(Config)
    #model = Text_Condition_Encoder(Config)
    #model = Text_CNN(Config)
    #model = Fast_Text(Config)
    Config.corpus = corpus
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-6)
    #optim = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    loss_fun = CrossEntropyLoss(size_average=False)

    best_micro_F1 = 0.0
    target_idx = 4
    for e_i in range(Config.epoch):
        print("\n----------------epoch: %d--------------------"%e_i)
        train(model, Config, loss_fun, optim, target_idx)
        micro_F1 = test_target(model, Config, loss_fun, target_idx)
        #micro_F1 = test_all_target(model, Config, loss_fun)
        if micro_F1 > best_micro_F1:
            best_micro_F1 = micro_F1
        print("micro: %f  best: %f"%(micro_F1, best_micro_F1))
        print("----------------epoch: %d--------------------\n"%e_i)

if __name__ == "__main__":
    np.random.seed(13)
    main()