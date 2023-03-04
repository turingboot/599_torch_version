import torch
from torch import nn


# Bi-LSTM encoder-decoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device="cpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]

        h_0 = torch.randn(2 * self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(2 * self.num_layers, batch_size, self.hidden_size).to(self.device)
        output, (h, c) = self.lstm(input_seq, (h_0, c_0))

        return h, c


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, input_seq, h, c):
        batch_size = input_seq.shape[0]
        input_seq = input_seq.view(batch_size, 1, self.input_size)
        output, (h, c) = self.lstm(input_seq, (h, c))
        pred = self.linear(output)  # pred(batch_size, 1, output_size)
        pred = pred[:, -1, :]

        return pred, h, c


class Seq2Seq(nn.Module):
    def __init__(self, channel_num, hidden_size, num_layers, output_size, batch_size, device="cpu"):
        super().__init__()
        self.output_size = output_size
        self.Encoder = Encoder(channel_num, hidden_size, num_layers, batch_size, device=device)
        self.Decoder = Decoder(channel_num, hidden_size, num_layers, output_size, batch_size)
        self.device = device

    def forward(self, input_seq):
        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2]
        h, c = self.Encoder(input_seq)
        outputs = torch.zeros(batch_size, seq_len, self.output_size).to(self.device)
        for t in range(seq_len):
            _input = input_seq[:, t, :]
            output, h, c = self.Decoder(_input, h, c)
            outputs[:, t, :] = output
        return outputs[:, -1, :]
