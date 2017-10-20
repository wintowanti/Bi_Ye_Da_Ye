#coding=UTF-8
from __future__ import print_function
import torch
from torch.autograd import Variable
from torch.nn.functional import softmax, dropout, relu
from torch.nn.init import xavier_normal

class Text_Condition_Encoder(torch.nn.Module):
    def __init__(self, config):
        super(Text_Condition_Encoder, self).__init__()
        self.config = config

        self.embedding_matrix = torch.nn.Embedding(config.voc_len, config.embedding_size)
        if config.embedding_matrix is not None:
            self.embedding_matrix.weight.data.copy_(torch.from_numpy(config.embedding_matrix))
        #self.embedding_matrix.weight.requires_grad = True
        self.target_lstm = torch.nn.LSTM(input_size=config.embedding_size, hidden_size=config.hidden_size, batch_first=True, bidirectional=False)
        self.tweet_lstms = []
        # for i in range(5):
        #     lstm =torch.nn.LSTM(input_size=config.embedding_size, hidden_size=config.hidden_size, batch_first=True, bidirectional=False)
        #     self.tweet_lstms.append(lstm)
        self.test =torch.nn.LSTM(input_size=config.embedding_size, hidden_size=config.hidden_size, batch_first=True, bidirectional=False)

        for lstm in self.tweet_lstms:
            initrange = 0.1
            #lstm.weight.data.uniform_(-initrange, initrange)
            xavier_normal(lstm.all_weights[0][0])
            xavier_normal(lstm.all_weights[0][1])
            #xavier_normal(lstm.all_weights[0][2])
            #xavier_normal(lstm.all_weights[0][3])
        #self.bi_lstm = torch.nn.GRU(input_size=config.embedding_size, hidden_size=config.hidden_size, batch_first=True, bidirectional=False)


        config.hidden_size *= 1
        self.fc_targets = []
        for i in range(5):
            self.fc_targets.append(torch.nn.Linear(config.hidden_size, config.class_size))

    def forward(self, text, target, target_idx):
        text = Variable(torch.from_numpy(text))
        batch_size = text.size(0)
        target = Variable(torch.from_numpy(target))

        text_em = self.embedding_matrix(text)
        target_em = self.embedding_matrix(target)

        output, (hn, cn) = self.target_lstm(target_em)
        zero_hn = Variable(torch.zeros([1, batch_size, self.config.hidden_size]), requires_grad=False)
        output, (hn, cn) = self.test(text_em,(zero_hn,cn))
        #output, (hn, cn) = self.test(text_em)
        #output, (hn, cn) = self.test(text_em)

        #hn = relu(hn)
        #hn = dropout(hn)
        #hn = torch.cat(hn, dim=1)
        hn = hn.view(-1, self.config.hidden_size)

        output1 = self.fc_targets[target_idx](hn)
        output2 = None

        return output1, output2
