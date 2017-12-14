#coding=UTF-8
from __future__ import print_function
import torch
from torch.autograd import Variable
from torch.nn.functional import softmax, dropout, relu, tanh

class Attention_Bi_LSTM_Condition(torch.nn.Module):
    def __init__(self, config):
        #torch.manual_seed(13)
        super(Attention_Bi_LSTM_Condition, self).__init__()
        self.config = config

        self.embedding_matrix = torch.nn.Embedding(config.voc_len, config.embedding_size)
        if config.embedding_matrix is not None:
            self.embedding_matrix.weight.data.copy_(torch.from_numpy(config.embedding_matrix))
        self.embedding_matrix.weight.requires_grad = False

        self.lstm = torch.nn.LSTM(input_size=config.embedding_size, hidden_size=config.hidden_size,
                                batch_first=True, bidirectional=True, dropout=0.3)

        self.target_hid = config.hidden_size
        self.lstm_target = torch.nn.LSTM(input_size=config.embedding_size, hidden_size=self.target_hid,
                                  batch_first=True, bidirectional=False, dropout=0.3)
        config.hidden_size *= 2

        self.attention = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size + self.target_hid, config.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_size, 1),
        )

        self.fc_target = torch.nn.Linear(config.hidden_size, config.class_size)
        # self.fc_targets = []
        # for i in range(5):
        #     self.fc_targets.append(torch.nn.Linear(config.hidden_size, config.class_size))

    def forward(self, text, target, target_idx):
        text = Variable(torch.from_numpy(text))
        target = Variable(torch.from_numpy(target))
        batch_size = text.size(0)
        text_em = self.embedding_matrix(text)
        text_em = dropout(text_em, 0.2)

        target_em = self.embedding_matrix(target)
        target_em = dropout(target_em, 0.2)

        #target_mean = torch.mean(target_em, dim=1, keepdim=True)
        _output, (target_hid, _cn) = self.lstm_target(target_em)
        target_hid = target_hid.view(-1, 1, self.target_hid)
        target_hid_rep = target_hid.repeat(1, self.config.fixed_len, 1)

        cn_target = _cn.repeat(2,1,1)
        hn = Variable(torch.zeros(cn_target.data.size()))

        output, hn = self.lstm(text_em, (hn,cn_target))

        join_hn = torch.cat([output, target_hid_rep], dim=2)
        ei = self.attention(join_hn)
        ei = ei.view(batch_size, -1)
        Ai = softmax(ei).view(batch_size, -1, 1).repeat(1,1,self.config.hidden_size)
        s = torch.sum((output * Ai), dim=1)
        s = dropout(s)
        output1 = self.fc_target(s)
        attention_metrix = softmax(ei).view(batch_size, -1, 1)
        output2 = None
        #output1 = self.fc_target1(hn)
        return output1, attention_metrix




if __name__ == "__main__":
    print("good")