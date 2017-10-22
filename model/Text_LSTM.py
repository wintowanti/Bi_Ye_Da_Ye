#coding=UTF-8
from __future__ import print_function
import torch
from torch.autograd import Variable
from torch.nn.functional import softmax, dropout, relu

class Text_LSTM(torch.nn.Module):
    def __init__(self, config):
        #torch.manual_seed(13)
        super(Text_LSTM, self).__init__()
        self.config = config

        self.embedding_matrix = torch.nn.Embedding(config.voc_len, config.embedding_size)
        if config.embedding_matrix is not None:
            self.embedding_matrix.weight.data.copy_(torch.from_numpy(config.embedding_matrix))
        self.embedding_matrix.weight.requires_grad = True
        self.bi_lstm = torch.nn.LSTM(input_size=config.embedding_size, hidden_size=config.hidden_size, batch_first=True, bidirectional=True)
        #self.bi_lstm = torch.nn.GRU(input_size=config.embedding_size, hidden_size=config.hidden_size, batch_first=True, bidirectional=False)


        config.hidden_size *= 2
        self.fc_target1 = torch.nn.Linear(config.hidden_size, config.class_size)
        self.fc_target2 = torch.nn.Linear(config.hidden_size, config.class_size)
        self.fc_target3 = torch.nn.Linear(config.hidden_size, config.class_size)
        self.fc_target4 = torch.nn.Linear(config.hidden_size, config.class_size)
        self.fc_target5 = torch.nn.Linear(config.hidden_size, config.class_size)
        self.fc_target6 = torch.nn.Linear(config.hidden_size, config.class_size)

    def forward(self, text, target, pair_idx):
        text = Variable(torch.from_numpy(text))
        batch_size = text.size(0)
        text_em = self.embedding_matrix(text)
        output, (hn, cn) = self.bi_lstm(text_em)
        #hn = torch.cat(hn, dim=1)
        hn = hn.view(-1, self.config.hidden_size)
        hn = relu(hn)
        hn = dropout(hn)
        # if pair_idx == 0:
        #     output1 = self.fc_target1(hn)
        #     output2 = self.fc_target2(hn)
        # elif pair_idx == 1:
        #     output1 = self.fc_target3(hn)
        #     output2 = self.fc_target4(hn)
        # else:
        #     output1 = self.fc_target5(hn)
        #     output2 = self.fc_target6(hn)

        # if pair_idx == 0:
        #     output1 = self.fc_target1(hn)
        #     output2 = self.fc_target2(hn)
        # elif pair_idx == 1:
        #     output1 = self.fc_target1(hn)
        #     output2 = self.fc_target3(hn)
        # else:
        #     output1 = self.fc_target2(hn)
        #     output2 = self.fc_target4(hn)
        output2 = None
        output1 = self.fc_target1(hn)
        return output1, output2




if __name__ == "__main__":
    print("good")
