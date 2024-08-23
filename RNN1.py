import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import __add__
from custom_lstm import BidirectionalRNN

GPU = True
device = torch.device("cuda" if torch.cuda.is_available() and GPU else "cpu")


class RNNAttention(nn.Module):
    def __init__(self, catagories, rate=16000, length=16000):
        super(RNNAttention, self).__init__()
        sr = rate
        iLen = length

        kernel_size1 = (5, 1)
        conv_padding = reduce(__add__,
                              [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size1[::-1]])
        self.zp1 = nn.ZeroPad2d(conv_padding)
        self.c1 = nn.Conv2d(128, 10, kernel_size1)
        self.bn1 = nn.BatchNorm2d(10)

        kernel_size2 = (5, 1)
        conv_padding = reduce(__add__,
                              [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size2[::-1]])
        self.zp2 = nn.ZeroPad2d(conv_padding)
        self.c2 = nn.Conv2d(10, 1, kernel_size2)
        self.bn2 = nn.BatchNorm2d(1)

        self.RNN1 = BidirectionalRNN(9, 50, 3)  # Use the custom LSTM submodule
        self.RNN2 = BidirectionalRNN(90, 64, 2)  # Use the custom LSTM submodule
        self.WQ = nn.Linear(128, 128)

        self.lin1 = nn.Linear(128, 64)
        self.lin2 = nn.Linear(64, catagories)

    def forward(self, x):
        b = x.shape[0]
        x = torch.unsqueeze(x, -1)
        x = self.zp1(x)
        x = self.c1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.zp2(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = torch.squeeze(x, -1).transpose(1, 2)
        x = torch.reshape(x, (b, 9, 9))
        x = self.RNN1(x)
        x = torch.reshape(x, (b, 10, 90))
        x = self.RNN2(x)

        x1 = x[:, -1, :]
        Q = self.WQ(x1)
        att = torch.einsum("bij,bj->bi", x, Q)
        att = torch.softmax(att, -1)
        x = torch.einsum("bji,bj->bi", x, att)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        return x
