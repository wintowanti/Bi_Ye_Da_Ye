#coding=UTF-8
from __future__ import print_function
import torch
from torch.autograd import Variable
from torch.nn.functional import softmax, dropout


class Text_CNN(torch.nn.Module):
    def __init__(self, config):
        #torch.manual_seed(13)
        super(Text_CNN, self).__init__()
        self.config = config
        sum_filter = 0

        self.embedding_matrix = torch.nn.Embedding(config.voc_len, config.embedding_size)
        if config.embedding_matrix is not None:
            self.embedding_matrix.weight.data.copy_(torch.from_numpy(config.embedding_matrix))
        self.embedding_matrix.weight.requires_grad = True

        window_size = 3
        small_filter = 100
        sum_filter += small_filter
        self.conv1_seq = torch.nn.Sequential(
            torch.nn.Conv1d(config.embedding_size, small_filter, kernel_size=window_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=config.fixed_len-window_size+1)
        )

        window_size = 4
        sum_filter += small_filter
        self.conv2_seq = torch.nn.Sequential(
            torch.nn.Conv1d(config.embedding_size, small_filter, kernel_size=window_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=config.fixed_len-window_size+1)
        )

        window_size = 5
        sum_filter += small_filter
        self.conv3_seq = torch.nn.Sequential(
            torch.nn.Conv1d(config.embedding_size, small_filter, kernel_size=window_size),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=config.fixed_len-window_size+1)
        )

        self.config.sum_filter = sum_filter
        #5 fc for each target
        self.fc_targets = []
        for i in range(5):
            self.fc_targets.append(torch.nn.Linear(sum_filter, config.class_size))

    def forward(self, text, target, target_idx):
        text = Variable(torch.from_numpy(text))
        batch_size = text.size(0)
        text_em = self.embedding_matrix(text)
        text_em = text_em.view([-1, self.config.fixed_len, self.config.embedding_size])
        text_em = text_em.permute(0, 2, 1)

        hn1 = self.conv1_seq(text_em)
        hn1 = hn1.view(batch_size, -1)

        hn2 = self.conv2_seq(text_em)
        hn2 = hn2.view(batch_size, -1)

        hn3 = self.conv3_seq(text_em)
        hn3 = hn3.view(batch_size, -1)

        hn = torch.cat([hn1, hn2, hn3], dim=1)

        output1 = self.fc_targets[target_idx](hn)
        output2 = None

        return output1, output2