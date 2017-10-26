#coding=UTF-8
from __future__ import print_function

import torch
from torch.autograd import Variable
from torch.nn.functional import softmax, dropout, relu
from torch.nn.init import xavier_normal

class LSTM_Bi_Condition_Encoder(torch.nn.Module):
    def __init__(self, config):
        super(LSTM_Bi_Condition_Encoder, self).__init__()
        self.config = config

        self.embedding_matrix = torch.nn.Embedding(config.voc_len, config.embedding_size)
        if config.embedding_matrix is not None:
            self.embedding_matrix.weight.data.copy_(torch.from_numpy(config.embedding_matrix))
        self.embedding_matrix.weight.requires_grad = False
        self.target_lstm = torch.nn.LSTM(input_size=config.embedding_size, hidden_size=config.hidden_size, batch_first=True, bidirectional=True)
        self.text_bi_lstm =torch.nn.LSTM(input_size=config.embedding_size, hidden_size=config.hidden_size, batch_first=True, bidirectional=True)

        config.hidden_size *= 2
        self.fc_targets = []
        for i in range(5):
            self.fc_targets.append(torch.nn.Linear(config.hidden_size, config.class_size))

    def forward(self, text, target, target_idx):
        text = Variable(torch.from_numpy(text))

        batch_size = text.size(0)
        target = Variable(torch.from_numpy(target))

        text_em = dropout(self.embedding_matrix(text), 0.2)
        target_em = dropout(self.embedding_matrix(target), 0.2)

        output, (hn, cn) = self.target_lstm(target_em)
        zero_hn = Variable(torch.zeros([2, batch_size, self.config.hidden_size/2]), requires_grad=False)
        output, (hn_bi, cn) = self.text_bi_lstm(text_em, (zero_hn, cn))

        hn = torch.cat(hn_bi, dim=1)

        hn = relu(hn)
        hn = dropout(hn)
        hn = hn.view(-1, self.config.hidden_size)

        output1 = self.fc_targets[target_idx](hn)
        output2 = None

        return output1, output2