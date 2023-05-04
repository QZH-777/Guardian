# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
'''Convolutional Neural Networks for Sentence Classification'''
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed, padding_idx=1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), 2)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)) # [batch_size, num_filters, seq_len-k+1, 1]
        x = x.squeeze(3) # [batch_size, num_filters, seq_len-k+1]
        x = F.max_pool1d(x, kernel_size=x.size(2)) # [batch_size, num_filters, 1]
        x = x.squeeze(2) # [batch_size, num_filters]
        return x

    def forward(self, x):
        out = self.embedding(x) # [batch_size, seq_len, embedding_size]
        out = out.unsqueeze(1) # [batch_size, 1, seq_len, embedding_size]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

model = Model(None)
model()
