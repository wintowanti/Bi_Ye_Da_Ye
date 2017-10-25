#coding=UTF-8
from __future__ import print_function
import torch
from torch.autograd import Variable
from torch.nn.functional import softmax, dropout, relu

class LSTM_Text_Target_Concat(torch.nn.Module):
    def __init__(self, config):
        #torch.manual_seed(13)
        super(LSTM_Text_Target_Concat, self).__init__()
        self.config = config

        self.embedding_matrix = torch.nn.Embedding(config.voc_len, config.embedding_size)
        if config.embedding_matrix is not None:
            self.embedding_matrix.weight.data.copy_(torch.from_numpy(config.embedding_matrix))
        self.embedding_matrix.weight.requires_grad = True
        self.text_lstm = torch.nn.LSTM(input_size=config.embedding_size, hidden_size=config.hidden_size, batch_first=True,
                                  bidirectional=False)
        self.target_lstm = torch.nn.LSTM(input_size=config.embedding_size, hidden_size=config.hidden_size, batch_first=True,
                                       bidirectional=False)

        config.hidden_size *= 2
        self.fc_target = torch.nn.Linear(config.hidden_size, config.class_size)

    def forward(self, text, target, pair_idx):
        text = Variable(torch.from_numpy(text))
        target = Variable(torch.from_numpy(target))

        batch_size = text.size(0)

        target_em = self.embedding_matrix(target)
        target_em = dropout(target_em, 0.2)

        text_em = self.embedding_matrix(text)
        text_em = dropout(text_em, 0.2)

        output, (hn1, cn) = self.text_lstm(text_em)
        output, (hn2, cn) = self.target_lstm(target_em)
        hn = torch.cat([hn1, hn2], dim=2)
        hn = hn.view(-1, self.config.hidden_size)
        hn = relu(hn)
        hn = dropout(hn)
        output2 = None
        output1 = self.fc_target(hn)
        return output1, output2




if __name__ == "__main__":
    print("good")