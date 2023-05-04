# coding: UTF-8
import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset
import pandas as pd
import jieba
import re
import config

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

""" 创建字典 """
def build_vocab(vocab_path):
    vocab = {}
    with open("./data/word2id.json", 'r', encoding='utf-8') as f:  # 词典
        word2id = json.loads(f.read())
    vocab['<unk>'] = 0
    vocab['<pad>'] = 1
    for key in word2id.keys():
        if key not in vocab:
            vocab[key] = len(vocab)
    return vocab # 是个字典

vocab = build_vocab(config.vocab_path)

class MyDataset(Dataset):
    def __init__(self, path, is_demo=False):
        content = pd.read_csv(path, dtype=str, header=None)
        content = np.array(content)
        sent_lists = content[:, 1] # 每个元素都是一个句子
        label_lists = content[:, 0] # 每个元素都是标签
        self.sent_lists = []
        self.label_lists = []
        for sent, label in zip(sent_lists, label_lists):
            sent_list = jieba.lcut(re.sub("[`!?:;\s+_,.$%^*(\"\')]+|[:：+—()?【】“”！，。？、~@#￥%…&*（）]+", " ", sent))
            sent_list = [x for x in sent_list if x != ' ']
            self.sent_lists.append(sent_list)
            if is_demo == True:
                self.label_lists.append(-1)
            else:
                self.label_lists.append(int(label))
    def __len__(self):
        # 返回的是句子的长度
        assert (len(self.label_lists) == len(self.sent_lists))
        return len(self.label_lists)

    def __getitem__(self, index):
        return self.sent_lists[index], self.label_lists[index]

def batch_process(batch):
    x_lists, y_lists = zip(*batch)
    PAD = vocab.get('<pad>')
    UNK = vocab.get('<unk>')

    batch_size = len(x_lists)

    x_tensor = torch.ones(batch_size, config.pad_size).long() * PAD  # 按照最长补齐
    for i, words in enumerate(x_lists):
        for j, word in enumerate(words):
            if j >= config.pad_size:
                break
            x_tensor[i][j] = vocab.get(word, UNK)  # 将词语转换成了词典中对应的序号（str——>int）
    y_tensor = torch.tensor(y_lists)

    return x_tensor, y_tensor

def make_word2id():
    content = pd.read_csv('./data/train.csv', delimiter=',', header=None, index_col=None)
    content = list(np.array(content)[:, 1:].tolist())
    print(content)
    word2id = {}
    for i in range(len(content)):
        for j, sen in enumerate(content[i]):
            out_list = jieba.lcut(re.sub("[`!?:;\s+_,.$%^*(\"\')]+|[:：+—()?【】“”！，。？、~@#￥%…&*（）]+", " ", sen))
            out_list = [x for x in out_list if x != ' ']
            for z in out_list:
                if z not in word2id:
                    word2id[z] = 0
                else:
                    word2id[z] += 1
    word2id_ = sorted(word2id.items(), key=lambda x: x[1], reverse=True)
    word2id = {x[0]: x[1] for x in word2id_}
    print(len(word2id))
    print(word2id)

    with open('./data/word2id1.json', 'w', encoding='utf-8') as fp:
        json.dump(word2id, fp)

