#coding=UTF-8
from __future__ import print_function
import torch
from torch.autograd import Variable
from torch.nn.functional import softmax, dropout, relu

class Bi_GRU(torch.nn.Module):
    def __init__(self, config):
        #torch.manual_seed(13)
        super(Bi_GRU, self).__init__()
        self.config = config

        self.embedding_matrix = torch.nn.Embedding(config.voc_len, config.embedding_size)
        if config.embedding_matrix is not None:
            self.embedding_matrix.weight.data.copy_(torch.from_numpy(config.embedding_matrix))
        self.embedding_matrix.weight.requires_grad = True
        self.gru = torch.nn.GRU(input_size=config.embedding_size, hidden_size=config.hidden_size,
                                batch_first=True, bidirectional=True, dropout=0.3)


        config.hidden_size *= 2

        self.fc_target = torch.nn.Linear(config.hidden_size, config.class_size)
        # self.fc_targets = []
        # for i in range(5):
        #     self.fc_targets.append(torch.nn.Linear(config.hidden_size, config.class_size))

    def forward(self, text, target, target_idx):
        text = Variable(torch.from_numpy(text))
        batch_size = text.size(0)
        text_em = self.embedding_matrix(text)
        text_em = dropout(text_em, 0.2)
        output, hn = self.gru(text_em)

        hn = torch.cat(hn, dim=1)
        hn = relu(hn)
        hn = dropout(hn)

        hn = hn.view(-1, self.config.hidden_size)
        output1 = self.fc_target(hn)
        output2 = None
        #output1 = self.fc_target1(hn)
        return output1, output2




if __name__ == "__main__":
    print("good")