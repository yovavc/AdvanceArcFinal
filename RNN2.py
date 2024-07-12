
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import __add__
import torch.nn.functional as F


# def get_melspec_model(iLen=None):
#     inp = L.Input((iLen,), name='input')
#     mel_spec = audioUtils.normalized_mel_spectrogram(inp)
#     melspecModel = Model(inputs=inp, outputs=mel_spec, name='normalized_spectrogram_model')
#     return melspecModel

GPU = True
device = torch.device("cuda" if torch.cuda.is_available() and GPU else "cpu")


class BidirectionalRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BidirectionalRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(2, self.hidden_dim)
        c0 = torch.zeros(2, self.hidden_dim)
        # Forward propagate RNN
        out, _ = self.rnn(x, (h0, c0))

        # Decode hidden state of last time step
        return out


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

        # x = Reshape((125, 80)) (x)
        # keras.backend.squeeze(x, axis)
        # self.RNN1 = BidirectionalRNN(81, 100).to(device)
        # self.RNN2 = BidirectionalRNN(100, 128).to(device)
        self.RNN1 = nn.LSTM(9, 50, 3, bidirectional=True, batch_first=True)
        self.RNN2 = nn.LSTM(90, 64, 2, bidirectional=True, batch_first=True)
        self.WQ = nn.Linear(128, 128)

        self.lin1 = nn.Linear(128, 64)
        self.lin2 = nn.Linear(64, 35)

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
        x, _ = self.RNN1(x)
        x = torch.reshape(x, (b, 10, 90))
        x, _ = self.RNN2(x)

        x1 = x[:, -1, :]
        Q = self.WQ(x1)
        att = torch.einsum("bij,bj->bi", x, Q)
        att = torch.softmax(att, -1)
        x = torch.einsum("bji,bj->bi", x, att)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        return x
