import torch
import torch.nn as nn


class BidirectionalRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BidirectionalRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # Forward propagate RNN
        out, _ = self.rnn(x, (h0, c0))
        return out
