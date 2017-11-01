# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

from data_helper import NLPCC_Corpus
from preprocess import init_embedding, fixed_length, stance2idx, metric
from utils import RedirectStdout
from config import Config
from copy import deepcopy
from model import Fast_Text, Text_CNN, LSTM_Text_Only, Text_GRU, Bi_GRU_CNN, Bi_GRU, Attention_Bi_GRU
from model import Attention_Bi_GRU_CNN, LSTM_Text_Target_Concat, lstm_condition_encode, LSTM_Condition_Bi_Encoder
from model import LSTM_Bi_Condition_Encoder

import torch
from torch.autograd import Variable
from torch.nn import NLLLoss,CrossEntropyLoss
import numpy as np

def train(model, config, loss_fun, optim, target_idx):
    model.train()
    corpus = config.corpus
    all_loss = 0.0
    sample_len = 0
    for id_, (idx_tweets, idx_target, stances, target_idx) in enumerate(corpus.iter_epoch(target_idx, "Train", batch_size=16)):
    #for id_, (idx_tweets, idx_target, stances, sentiments, target_idx) in enumerate(corpus.iter_all_train_target(batch_size=30, is_random=False)):
        np_idx_tweets = fixed_length(idx_tweets, Config.fixed_len)
        np_idx_targets = fixed_length(idx_target, len(idx_target[0]))
        sample_len += len(idx_tweets)

        var_s1 = Variable(torch.from_numpy(stance2idx(stances)))
        output1, output2 = model(np_idx_tweets, np_idx_targets, target_idx)

        loss = loss_fun(output1, var_s1)
        all_loss += loss.data.sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
    print("-----------train mean loss: ", all_loss/sample_len)
    return model.state_dict()

def test(model, config, loss_fun, flag, target_idx):
    model.eval()
    corpus = config.corpus
    all_loss = 0.0
    sample_len = 0
    y_true_stances = np.array([])
    y_pred_stances = np.array([])

    for id_, (idx_tweets,idx_target, stances, target_idx) in enumerate(corpus.iter_epoch(target_idx, flag)):
        np_idx_tweets = fixed_length(idx_tweets, Config.fixed_len)
        np_idx_targets = fixed_length(idx_target, len(idx_target[0]))
        sample_len += len(stances)

        y_true_stances = np.append(y_true_stances, stance2idx(stances))

        var_s1 = Variable(torch.from_numpy(stance2idx(stances)))
        output1, output2 = model(np_idx_tweets, np_idx_targets, target_idx)

        y_pred_stances = np.append(y_pred_stances, torch.max(output1, dim=1, keepdim=False)[1].data.numpy())
        loss = loss_fun(output1, var_s1)
        all_loss += loss.data.sum()
    f1_target1 = metric(y_true=y_true_stances, y_pred=y_pred_stances, name="target1")
    print(flag+" mean loss: ", all_loss/sample_len)
    print("f1: %f"%f1_target1)
    return f1_target1, y_true_stances, y_pred_stances

def get_model(config):
    #model = LSTM_Condition_Bi_Encoder(config)
    #model = LSTM_Bi_Condition_Encoder(config)
    #model = Attention_Bi_GRU_CNN(config)
    #model = Attention_Bi_GRU(config)
    #model = Bi_GRU_CNN(config)
    #model = Text_GRU(config)
    #model = Bi_GRU(config)
    #model = Attention_Bi_GRU_CNN(config)
    model = Text_CNN(config)
    #model = Fast_Text(config)
    #model = LSTM_Text_Only(config)
    #model = LSTM_Text_Target_Concat(config)
    return model

def test_all_target(Config):
    y_true_stances = np.array([])
    y_pred_stances = np.array([])
    for target_idx in range(5):
        target_idx = 0
        model = get_model(Config)
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005, weight_decay=1e-6)
        loss_fun = CrossEntropyLoss(size_average=False)
        best_micro_F1 = 0.0
        best_model_dict = None
        for e_i in range(Config.epoch):
            print("\n----------------epoch: %d--------------------" % e_i)
            model_dict = train(model, Config, loss_fun, optim, target_idx)
            micro_F1, t, p = test(model, Config, loss_fun, "Dev", target_idx)
            if micro_F1 > best_micro_F1:
                best_micro_F1 = micro_F1
                best_model_dict = deepcopy(model_dict)
            print("micro: %f  best dev : %f" % (micro_F1, best_micro_F1))
            print("----------------epoch: %d--------------------\n" % e_i)
        model.load_state_dict(best_model_dict)
        file_name = "output/" + "nlpcc_" + model.__class__.__name__ + ".res"
        with open(file_name, "a+") as f:
            with RedirectStdout(f):
                print("\n------target: %s----------" % (Config.corpus.targets[target_idx].encode("UTF-8")))
                micro_F1, true_stances, pred_stances, = test(model, Config, loss_fun, "Test", target_idx)
        y_true_stances = np.append(y_true_stances, true_stances)
        y_pred_stances = np.append(y_pred_stances, pred_stances)
    with open(file_name, "a+") as f:
        with RedirectStdout(f):
            metric(y_pred_stances, y_true_stances, "Average")
    return

def test_target(model, Config, loss_fun, target_idx):
    # f1_targets = []
    # y_true_stances = np.array([])
    # y_pred_stances = np.array([])
    # print("target: %s"%(Config.corpus.targets[target_idx]))
    # f1, t, p = test(model, Config, loss_fun, "Train", target_idx)
    # f1, t, p = test(model, Config, loss_fun, "Test", target_idx)
    # f1_targets.append(f1)
    # y_true_stances = np.append(y_true_stances, t) # y_pred_stances = np.append(y_pred_stances, p)
    # micro_f1 = metric(y_true_stances, y_pred_stances, "micro-F1")
    #return micro_f1
    pass

def main():
    corpus = NLPCC_Corpus(Config.nlpcc_data_path)
    Config.embedding_size = 200
    embedding_matrix = init_embedding(corpus.dictionary.word2idx, Config.embedding_size, Config.weibo_embedding_path)
    Config.voc_len = len(corpus.dictionary)
    Config.embedding_matrix = embedding_matrix
    Config.corpus = corpus
    Config.fixed_len = 50
    test_all_target(Config)
    # optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-6)
    # #optim = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    # loss_fun = CrossEntropyLoss(size_average=False)
    #
    # best_micro_F1 = 0.0
    # target_idx = 0
    # for e_i in range(Config.epoch):
    #     print("\n----------------epoch: %d--------------------"%e_i)
    #     train(model, Config, loss_fun, optim, target_idx)
    #     micro_F1 = test_target(model, Config, loss_fun, target_idx)
    #     #micro_F1 = test_all_target(model, Config, loss_fun)
    #     if micro_F1 > best_micro_F1:
    #         best_micro_F1 = micro_F1
    #     print("micro: %f  best: %f"%(micro_F1, best_micro_F1))
    #     print("----------------epoch: %d--------------------\n"%e_i)

if __name__ == "__main__":
    np.random.seed(13)
    main()