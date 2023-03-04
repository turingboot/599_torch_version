import torch
from torch import nn
from torch.autograd import Variable


class SMP_LSTM(nn.Module):
    def __init__(self, input_timestamp=24, input_dim=14, hidden_size=50, out_size=1, device='cpu'):
        super().__init__()
        self.input_timestamp = input_timestamp
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_timestamp * input_dim, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)
        self.hidden = (torch.zeros(1, 1, hidden_size, device=device), torch.zeros(1, 1, hidden_size, device=device))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq), -1))
        return pred[-1]


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, device=None):
        super().__init__()
        self.device = device
        self.num_classes = num_classes  # output size
        self.num_layers = num_layers  # number of recurrent layers in the lstm
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected
        self.fc_2 = nn.Linear(128, num_classes)  # fully connected last layer
        self.relu = nn.ReLU()

    def forward(self, x):
        # hidden state
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device))
        # cell state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device))
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # (input, hidden, and internal state)
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)  # final output
        return out
