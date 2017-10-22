# -*- coding: utf-8 -*-
from __future__ import print_function
from data_helper import Corpus
from preprocess import init_embedding, fixed_length, stance2idx, metric
from config import Config
import torch
from torch.autograd import Variable
from torch.nn import NLLLoss,CrossEntropyLoss
import numpy as np

target1 = "Donald Trump"
target2 = "Hilary Clinton"
def train(model, config, epoch, loss_fun, optim, target1, target2, pair_idx):
    model.train()
    corpus = config.corpus
    all_loss = 0.0
    sample_len = 0
    for id_, (idx_tweets, s1, s2, pair_idx) in enumerate(corpus.iter_train_epoch()):
    #for id_, (idx_tweets, s1, s2, pair_idx) in enumerate(corpus.iter_epoch(target1, target2, flag="Train", pair_idx=pair_idx)):
        np_idx_tweets = fixed_length(idx_tweets, Config.fixed_len)
        sample_len += len(s1)

        var_s1 = Variable(torch.from_numpy(stance2idx(s1)))
        var_s2 = Variable(torch.from_numpy(stance2idx(s2)))
        output1, output2 = model(np_idx_tweets, pair_idx)

        #loss = loss_fun(output1, var_s1) + loss_fun(output2, var_s2)
        #train - independed
        loss = loss_fun(output1, var_s1)
        all_loss += loss.data.sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
    print("-----------train mean loss: ", all_loss/sample_len)

def test(model, config, loss_fun, flag, target1, target2, pair_idx):
    model.eval()
    corpus = config.corpus
    all_loss = 0.0
    sample_len = 0
    y_true_t1 = np.array([])
    y_pred_t1 = np.array([])

    y_true_t2 = np.array([])
    y_pred_t2 = np.array([])

    for id_, (idx_tweets, s1, s2, pair_idx) in enumerate(corpus.iter_epoch(target1, target2, flag, pair_idx, batch_size=64)):
        np_idx_tweets = fixed_length(idx_tweets, Config.fixed_len)
        sample_len += len(s1)

        y_true_t1 = np.append(y_true_t1, stance2idx(s1))
        y_true_t2 = np.append(y_true_t2, stance2idx(s2))

        var_s1 = Variable(torch.from_numpy(stance2idx(s1)))
        var_s2 = Variable(torch.from_numpy(stance2idx(s2)))
        output1, output2 = model(np_idx_tweets, pair_idx)

        y_pred_t1 = np.append(y_pred_t1, torch.max(output1, dim=1, keepdim=False)[1].data.numpy())
        y_pred_t2 = np.append(y_pred_t2, torch.max(output2, dim=1, keepdim=False)[1].data.numpy())
        loss = loss_fun(output1, var_s1) + loss_fun(output2, var_s2)
        all_loss += loss.data.sum()
    f1_target1 = metric(y_true=y_true_t1, y_pred=y_pred_t1, name="target1")
    f1_target2 = metric(y_true=y_true_t2, y_pred=y_pred_t2, name="target2")
    print(flag+" mean loss: ", all_loss/sample_len)
    average_f1 = (f1_target1+f1_target2)/2
    print("%s average f1 score %f"%(flag, average_f1))
    return f1_target1

def main():
    corpus = Corpus(Config.data_path)
    embedding_matrix = init_embedding(corpus.dictionary.word2idx, Config.embedding_size, Config.embedding_path)
    Config.voc_len = len(corpus.dictionary)
    Config.embedding_matrix = embedding_matrix
    #lstm_model = Text_LSTM(Config)
    lstm_model = Text_CNN(Config)
    Config.corpus = corpus
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, lstm_model.parameters()), lr=0.001)
    #optim = torch.optim.Adagrad(filter(lambda p: p.requires_grad, lstm_model.parameters()), lr=0.01)
    loss_fun = CrossEntropyLoss(size_average=False)

    best_f1_socre = 0.0
    best_average_socre = 0.0
    for e_i in range(Config.epoch):
        print("\n----------------epoch: %d--------------------"%e_i)
        pair_idx = 0
        train(lstm_model, Config, e_i, loss_fun, optim, corpus.pair_names[pair_idx][0], corpus.pair_names[pair_idx][1], pair_idx)
        f1 = test(lstm_model, Config, loss_fun, "Dev",corpus.pair_names[pair_idx][0], corpus.pair_names[pair_idx][1], pair_idx)
        #test(lstm_model, Config, loss_fun, "Test")
        if f1 + 0.01 < best_f1_socre and e_i > 50:
            break
        if f1 > best_f1_socre:
            best_f1_socre = f1

        average_f1_score = 0.0
        print("----------------Test--------------------")
        # for pair_idx, (target1, target2) in enumerate(corpus.pair_names):
        #     f1_socre = test(lstm_model, Config, loss_fun, "Test", target1, target2, pair_idx)
        #     print("pair(%s, %s) f1_score: %f"%(target1, target2, f1_socre))
        #     average_f1_score += f1_socre
        # average_f1_score /= 3.0

        if average_f1_score > best_average_socre: best_average_socre = average_f1_score
        print("average f1:score: %f   best: %f"%(average_f1_score, best_average_socre))
        print("----------------epoch: %d--------------------\n"%e_i)

if __name__ == "__main__":
    np.random.seed(13)
    main()
