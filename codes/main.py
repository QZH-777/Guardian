# coding: UTF-8
import time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from train_eval import train, init_network, test, demo
from importlib import import_module
import argparse
import pandas as pd
from TextCNN import Model
import config
from utils import vocab, MyDataset, batch_process

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--is_test', default=False, action='store_true', help='ture or false')
opt = parser.parse_args()

if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    print("Loading data...")
    config.vocab_size = len(vocab)
    trainDataSet = MyDataset(config.train_path)
    train_iter = DataLoader(trainDataSet, batch_size=config.batch_size, shuffle=True, collate_fn=batch_process)
    testDataSet = MyDataset(config.test_path)
    test_iter = DataLoader(testDataSet, batch_size=config.batch_size, shuffle=True, collate_fn=batch_process)
    validDataSet = MyDataset(config.valid_path)
    valid_iter = DataLoader(validDataSet, batch_size=config.batch_size, shuffle=True, collate_fn=batch_process)

    # train
    model = Model(config).to(config.device)
    if opt.is_test == False:
        init_network(model)
        print(model.parameters)
        train(config, model, train_iter, valid_iter, test_iter)
        test(config, model, test_iter)
    else:
        demoDataSet = MyDataset(config.demo_path, is_demo=True)
        demo_iter = DataLoader(demoDataSet, batch_size=config.batch_size, shuffle=False, collate_fn=batch_process)
        ans = demo(config, model, demo_iter)
        print(ans)
        w = pd.DataFrame(ans, columns=None)
        w.to_csv('./data/demo_sentiment.csv', header=None, index=None, encoding='utf-8')
