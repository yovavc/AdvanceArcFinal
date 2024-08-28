
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import __add__
import torch.nn.functional as F
from custom_lstm import BidirectionalRNN



GPU = True
device = torch.device("cuda" if torch.cuda.is_available() and GPU else "cpu")



class RNNAttention(nn.Module):
    def __init__(self, catagories):
        super(RNNAttention, self).__init__()

        kernel_size1 = (7, 5)
        conv_padding = reduce(__add__,
                              [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size1[::-1]])
        self.zp1 = nn.ZeroPad2d(conv_padding)
        self.c1 = nn.Conv2d(128, 64, kernel_size1)
        self.bn1 = nn.BatchNorm2d(64)

        kernel_size2 = (5, 3)
        conv_padding = reduce(__add__,
                              [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size2[::-1]])
        self.zp2 = nn.ZeroPad2d(conv_padding)
        self.c2 = nn.Conv2d(64, 32, kernel_size2)
        self.bn2 = nn.BatchNorm2d(32)

        kernel_size3 = (5, 1)
        conv_padding = reduce(__add__,
                              [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size3[::-1]])
        self.zp3 = nn.ZeroPad2d(conv_padding)
        self.c3 = nn.Conv2d(32, 16, kernel_size3)
        self.bn3 = nn.BatchNorm2d(16)

        self.up = nn.Linear(16, 32)
        # x = Reshape((125, 80)) (x)
        # keras.backend.squeeze(x, axis)
        # self.RNN1 = BidirectionalRNN(81, 100).to(device)
        # self.RNN2 = BidirectionalRNN(100, 128).to(device)
        self.RNN1 = BidirectionalRNN(81, 64, 3)
        self.RNN2 = BidirectionalRNN(64, 64, 3)
        self.WQ1 = nn.Linear(32, 32)
        self.WQ2 = nn.Linear(128, 128)
        self.RD = nn.Linear(128, 127)

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

        x = self.zp3(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = torch.squeeze(x, -1).transpose(1,2)
        x = self.up(x)
        x = x.transpose(1,2)

        x = self.RNN1(x)

        x1 = x[:, :, -1]
        Q = self.WQ1(x1)
        att = torch.einsum("bji,bj->bi", x, Q)
        att = torch.softmax(att, -1)
        att = torch.einsum("bij,bj->bi", x, att)

        x = self.RD(x)
        att = att.unsqueeze(-1)
        x = torch.cat((x, att), 2)

        # x = self.DN(x)
        x = torch.reshape(x, (b, 64, 64))
        x = self.RNN2(x)
        x1 = x[:, -1, :]

        Q = self.WQ2(x1)
        att = torch.einsum("bij,bj->bi", x, Q)
        att = torch.softmax(att, -1)
        x = torch.einsum("bji,bj->bi", x, att)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        return x
