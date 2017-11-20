#coding=UTF-8
from __future__ import print_function
import torch
from torch.autograd import Variable
from torch.nn.functional import softmax, dropout, relu, tanh

class Attention_Bi_GRU_CNN(torch.nn.Module):
    def __init__(self, config):
        #torch.manual_seed(13)
        super(Attention_Bi_GRU_CNN, self).__init__()
        self.config = config

        self.embedding_matrix = torch.nn.Embedding(config.voc_len, config.embedding_size)
        if config.embedding_matrix is not None:
            self.embedding_matrix.weight.data.copy_(torch.from_numpy(config.embedding_matrix))
        self.embedding_matrix.weight.requires_grad = True
        self.gru = torch.nn.GRU(input_size=config.embedding_size, hidden_size=config.hidden_size,
                                batch_first=True, bidirectional=True, dropout=0.3)

        config.hidden_size *= 2

        self.attention = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size+config.embedding_size, config.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_size, 1),
        )

        self.filter_num = filter_num = 50
        self.cnn2_seq = self.cnn_sequences(config.hidden_size, 3, filter_num)
        self.cnn3_seq = self.cnn_sequences(config.hidden_size, 4, filter_num)
        self.cnn4_seq = self.cnn_sequences(config.hidden_size, 2, filter_num)

        self.fc_target = torch.nn.Linear(filter_num*3, config.class_size)
        # self.fc_targets = []
        # for i in range(5):
        #     self.fc_targets.append(torch.nn.Linear(filter_num*3, config.class_size))


    def cnn_sequences(self,embedding_size, window_size, filter_num):
        cnn_seq = torch.nn.Sequential(
            torch.nn.Conv1d(embedding_size, filter_num, kernel_size=window_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=self.config.fixed_len - window_size + 1)
        )
        return cnn_seq

    def forward(self, text, target, target_idx):
        text = Variable(torch.from_numpy(text))
        target = Variable(torch.from_numpy(target))
        batch_size = text.size(0)
        text_em = self.embedding_matrix(text)
        text_em = dropout(text_em, 0.2)

        target_em = self.embedding_matrix(target)
        target_em = dropout(target_em, 0.2)

        target_mean = torch.mean(target_em, dim=1, keepdim=True)
        target_mean = target_mean.repeat(1, self.config.fixed_len, 1)

        output, hn = self.gru(text_em)

        join_hn = torch.cat([output, target_mean], dim=2)
        ei = self.attention(join_hn)
        ei = ei.view(batch_size, -1)
        Ai = softmax(ei).view(batch_size, -1, 1).repeat(1,1,self.config.hidden_size)

        output = output * Ai * self.config.fixed_len*1.0
        output = output.permute(0, 2, 1)

        h1 = self.cnn2_seq(output)
        h2 = self.cnn2_seq(output)
        h3 = self.cnn2_seq(output)

        hn = torch.cat([h1, h2, h3], dim=1)

        hn = hn.squeeze()
        hn = relu(hn)
        hn = dropout(hn)

        hn = hn.view(-1, self.filter_num * 3)
        output1 = self.fc_target(hn)
        output2 = None
        return output1, output2




if __name__ == "__main__":
    pass