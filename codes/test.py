import torch
from transformers import BertForSequenceClassification, BertConfig, AutoTokenizer

class Mymodel(torch.nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.linear1 = torch.nn.Linear(2, 2)

    def forward(self, x):
        x = self.linear1(x)
        return x



x = torch.rand([3,2], dtype=torch.float)
model = Mymodel()
y = model(x)
print(y)