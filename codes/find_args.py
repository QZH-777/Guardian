import numpy as np
import time
import torch
from train_eval import train, init_network
from importlib import import_module
from utils import build_dataset, build_iterator


"""
需训练参数有：batch_size, embedding_size, num_filters, learning_rate, dropout
(1)batch_size: 可行解域为[64, 256]，且设置该值为16的倍数。对应编码4位二进制 * 16 + 64
(2)embedding_size: 可行解域为[64, 256]，且设置该值为16的倍数。对应编码4位二进制 * 16 + 64
(3)num_filters: 可行解域为[64, 256]，且设置该值为16的倍数。对应编码4位二进制 * 16 + 64
(4)learning_rate: 可行解域为[e-4, 6e-4]，且设置该值为e-4的倍数。对应编码3位二进制 * e-4
(5)dropout: 可行解域为[0.1, 0.6]，且设置该值为0.1的倍数。对应编码3位二进制 * 0.1
因此总编码为 1111 | 1111 | 1111 | 111 | 111，一共18位。
1-4位决定batch_size的值，5-9位决定embedding_size的值，10-14位决定num_filters的值，15-18位决定learning_rate的值，19-22位决定dropout的值
"""
def code_split(code):
    code = np.array(code, dtype=str)
    s = ''.join(code)
    return int(s[:4],2) * 16 + 64, int(s[4:8],2) * 16 + 64, int(s[8:12],2) * 16 + 64, \
           int(s[12:15],2) * np.exp(-4), int(s[15:],2) * 0.1

answer = {} # 记录已经算出的结果

w, g = 50, 100 # w为初始种群的个数，g为进化的代数
init_grop = []
np.random.seed(0)
# 生成初始种群
for i in range(w):
    solution = np.random.randint(0, 2, 18)
    solution = np.array(solution, dtype=str)
    init_grop.append(solution)
init_grop = np.array(init_grop)
best_score = 0
best_var = 0
for k in range(g):
    A = np.copy(init_grop) # A为进行交叉操作后的种群
    for i in range(0, w, 2):
        ch1 = np.zeros((9,))
        ch1[0] = np.random.rand()
        for j in range(1, 9):
            ch1[j] = 4 * ch1[j - 1] * (1 - ch1[j - 1]) # 产生混沌序列
        ch1 = ch1 * 100 % 18
        for m in ch1:
            m = int(m)
            tmp = A[i][m]
            A[i][m] = A[i + 1][m]
            A[i + 1][m] = tmp
    poss = np.random.rand(w)
    by = []
    B = [] # B是变异操作后的种群
    for p_index, p_value in enumerate(poss):
        if p_value < 0.2: # 生成进行变异操作的个体
            by.append(p_index)
    if by != []:
        len_by = len(by)
        ch2 = np.zeros((9, ))
        ch2[0] = np.random.rand()
        for j in range(1, 9):
            ch2[j] = 4 * ch2[j - 1] * (1 - ch2[j - 1]) # 生成混沌序列
        B = [[] for _ in range(len_by)]
        for t in range(len_by):
            index = by[t]
            B[t] = init_grop[index]
            place = int(ch2[t] * 100 % 18)
            B[t][place] = 1 if np.random.rand() > 0.5 else 0 # 变异操作

    if B != []:
        B = np.array(B)
        G = np.vstack((A, B, init_grop))
    else:
        G = np.vstack((A, init_grop))

    best = []
    for index, example in enumerate(G):
        code = np.array(example, dtype=str)
        s = ''.join(code)
        if s in answer:
            continue
        batch_size, embedding_size, num_filters, learning_rate, dropout = code_split(example)
        batch_size = min(batch_size, 256)
        batch_size = max(64, batch_size)
        embedding_size = min(embedding_size, 300)
        embedding_size = max(64, embedding_size)
        num_filters = min(num_filters, 256)
        num_filters = max(64, num_filters)
        learning_rate = min(learning_rate, 6 * np.exp(-4))
        learning_rate = max(np.exp(-4), learning_rate)
        dropout = min(dropout, 0.6)
        dropout = max(0.1, dropout)
        dataset = 'THUCNews'  # 数据集
        args = [batch_size, embedding_size, num_filters, learning_rate, dropout]
        # 预训练模型：搜狗新闻:embedding_SougouNews.npz ; 腾讯:embedding_Tencent.npz
        # 随机初始化: random
        embedding = 'embedding_SougouNews.npz'
        x = import_module('models.TextCNN')
        config = x.Config(dataset, embedding, args)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        vocab, train_data, dev_data, test_data = build_dataset(config, False)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        # train
        config.n_vocab = len(vocab)
        model = x.Model(config).to(config.device)
        acc = train(config, model, train_iter, dev_iter, test_iter)
        answer[s] = acc
        best.append([index, acc]) # best存储该个体的下标和准确度
    best = sorted(best, key=lambda x:x[1], reverse=True)[:w] # 根据准确度进行排序，并选择前w个
    if best[0][1] > best_score: # 记录最优个体
        best_score = best[0][1]
        best_var = G[best[0][0]]
        print(best_score)
        print(best_var)
        print('############################')
    else: # 如果种群不再进化
        break
    for i, j in enumerate(best): # 生成下一个种群
        init_grop[i] = G[j[0]]

print(best_score, best_var)