import torch

"""配置参数"""
train_path = './data/train.csv' # 训练集
valid_path = './data/valid.csv'  # 验证集
test_path = './data/test.csv'  # 测试集
vocab_path = './data/word2id.json' # 词表
save_path = './model_sen' # 模型训练结果
demo_path = './data/COVID_vaccine.csv' # 待预测的未知数据
demo_label_path = './data/demo_sentiment.csv' # 预测结果

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

dropout = 0.3
require_improvement = 1000 # 若超过2000batch效果还没提升，则提前结束训练

num_epochs = 100 # epoch数
batch_size = 64 # mini-batch大小
pad_size = 32  # 每句话处理成的长度(短填长切)
learning_rate = 1e-4 # 学习率
embed =  256 # 字向量维度
vocab_size = 0 # 词典大小

filter_sizes = (2, 3, 4) # 卷积核尺寸
num_filters = 256 # 卷积核数量(channels数)
