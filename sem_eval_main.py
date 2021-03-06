# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

from data_helper import Seme_Corpus
from copy import deepcopy
from preprocess import init_embedding, fixed_length, stance2idx, metric
from config import Config
from model import Fast_Text, Text_CNN, LSTM_Text_Only, Text_GRU, Bi_GRU_CNN, Bi_GRU, Attention_Bi_GRU
from model import Attention_Bi_GRU_CNN, LSTM_Text_Target_Concat, LSTM_Condition_Encoder, LSTM_Condition_Bi_Encoder
from model import LSTM_Bi_Condition_Encoder
from model import Bi_LSTM, Attention_Bi_LSTM, Attention_Bi_LSTM_Condition
from utils import RedirectStdout, get_nowtime_str, judge

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
    #judge()
    print("-----------train mean loss: ", all_loss/sample_len)
    return model.state_dict()

def test(model, config, loss_fun, flag, target_idx, print_attention=False):
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
        output1, attention_matrix = model(np_idx_tweets, np_idx_targets, target_idx)

        y_pred_stances = np.append(y_pred_stances, torch.max(output1, dim=1, keepdim=False)[1].data.numpy())
        #y_pred_sentiments = np.append(y_pred_sentiments, torch.max(output2, dim=1, keepdim=False)[1].data.numpy())
        loss = loss_fun(output1, var_s1)
        all_loss += loss.data.sum()

        if print_attention == True:
            print_attention_fun(config, idx_tweets, stances, y_pred_stances, attention_matrix.data.numpy())
    f1_target1 = metric(y_true=y_true_stances, y_pred=y_pred_stances, name="target1")
    #f1_target2 = metric(y_true=y_true_sentiments, y_pred=y_pred_sentiments, name="target2")
    print(flag+" mean loss: ", all_loss/sample_len)
    print("f1: %f"%f1_target1)
    return f1_target1, y_true_stances, y_pred_stances

def print_attention_fun(config, id_tweets, stance, p_stance, attention_matrix):
    idx2word = config.corpus.dictionary.idx2word
    for id_tweet, s, ps, attention_weight in zip(id_tweets, stance, p_stance, attention_matrix):
        print("|".join([idx2word[id] for id in id_tweet]))
        print("|".join([ '{:.3f}'.format(float(a)) for a in attention_weight]))
        print(s)
        print(ps)
    pass

def get_model(config):
    #model = LSTM_Condition_Bi_Encoder(config)
    #model = LSTM_Bi_Condition_Encoder(config)
    #model = Attention_Bi_GRU_CNN(config)
    #model = Attention_Bi_GRU(config)
    #model = Bi_GRU_CNN(config)
    #model = Text_GRU(config)
    #model = Bi_GRU(config)
    #model = Attention_Bi_GRU_CNN(config)
    #model = Text_CNN(config)
    ##model = Fast_Text(config)
    #model = LSTM_Condition_Encoder(config)
    #model = LSTM_Text_Only(config)
    #model = LSTM_Text_Target_Concat(config)
    #model = Attention_Bi_LSTM(config)
    model = Attention_Bi_LSTM_Condition(config)
    #model = AT
    return model

def test_all_target(Config):
    y_true_stances = np.array([])
    y_pred_stances = np.array([])
    for target_idx in range(5):
        target_idx = 3
        model = get_model(Config)
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-6)
        loss_fun = CrossEntropyLoss(size_average=False)
        best_micro_F1 = 0.0
        best_model_dict = None
        for e_i in range(Config.epoch):
            print("\n----------------epoch: %d--------------------"%e_i)
            model_dict = train(model, Config, loss_fun, optim, target_idx)
            micro_F1, t, p= test(model, Config, loss_fun, "Dev", target_idx)
            if micro_F1 > best_micro_F1:
                best_micro_F1 = micro_F1
                best_model_dict = deepcopy(model_dict)
            print("micro: %f  best dev : %f"%(micro_F1, best_micro_F1))
            print("----------------epoch: %d--------------------\n"%e_i)
        model.load_state_dict(best_model_dict)
        file_name = "output/"+"visual_attention_sem_eval_"+model.__class__.__name__+".res"
        with open(file_name, "a+") as f:
            with RedirectStdout(f):
                print("\n------target: %s----------"%(Config.corpus.targets[target_idx]))
                micro_F1, true_stances, pred_stances, = test(model, Config, loss_fun, "Test", target_idx, print_attention=True)
        y_true_stances = np.append(y_true_stances, true_stances)
        y_pred_stances = np.append(y_pred_stances, pred_stances)
        with open(file_name, "a+") as f:
            with RedirectStdout(f):
                print("Time: %s"%get_nowtime_str())
                metric(y_pred_stances, y_true_stances, "Average")
    return

def test_target(model, Config, loss_fun, flag, target_idx):
    pass

def main():
    Config.fixed_len = 30
    Config.embedding_size = 100
    Config.hidden_size = 64
    corpus = Seme_Corpus(Config.seme_data_path)
    embedding_matrix = init_embedding(corpus.dictionary.word2idx, Config.embedding_size, Config.embedding_path)
    Config.voc_len = len(corpus.dictionary)
    Config.corpus = corpus
    Config.embedding_matrix = embedding_matrix
    #model = LSTM_Condition_Bi_Encoder(Config)
    #model = LSTM_Bi_Condition_Encoder(Config)
    #model = Attention_Bi_GRU_CNN(Config)
    #model = Attention_Bi_GRU(Config)
    #model = Bi_GRU_CNN(Config)
    #model = Text_GRU(Config)
    #model = Bi_GRU(Config)
    #model = Attention_Bi_GRU_CNN(Config)
    #model = Text_CNN(Config)
    ##model = Fast_Text(Config)
    #model = LSTM_Text_Only(Config)
    #model = LSTM_Text_Target_Concat(Config)
    #Config.corpus = corpus
    #optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-6)
    #optim = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    test_all_target(Config)

if __name__ == "__main__":
    np.random.seed(13)
    main()