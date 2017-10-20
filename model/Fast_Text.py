#coding=UTF-8
from __future__ import print_function
import torch
from torch.autograd import Variable
from torch.nn.functional import softmax, dropout, relu

class Fast_Text(torch.nn.Module):
    def __init__(self, config):
        #torch.manual_seed(13)
        super(Fast_Text, self).__init__()
        self.config = config

        self.embedding_matrix = torch.nn.Embedding(config.voc_len, config.embedding_size)
        if config.embedding_matrix is not None:
            self.embedding_matrix.weight.data.copy_(torch.from_numpy(config.embedding_matrix))
        self.embedding_matrix.weight.requires_grad = True

        config.last_feature_size = 200
        self.fc_targets = []
        for i in range(5):
            self.fc_targets.append(torch.nn.Linear(config.last_feature_size, config.class_size))

    def forward(self, text, target, target_idx):
        batch_size = text.shape[0]
        text = Variable(torch.from_numpy(text))
        target = Variable(torch.from_numpy(target))
        text_em = self.embedding_matrix(text)
        target_em = self.embedding_matrix(target)

        mean_text = torch.mean(text_em, 1)
        mean_target = torch.mean(target_em, 1)
        hn = torch.cat([mean_text, mean_target], dim=1)
        hn = relu(hn)
        hn = dropout(hn)
        hn = hn.view(-1, self.config.last_feature_size)
        output1 = self.fc_targets[target_idx](hn)
        output2 = None

        return output1, output2