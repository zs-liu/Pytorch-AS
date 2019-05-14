import torch
import torch.nn as nn


class SequentialAttention(nn.Module):
    def __init__(self, hidden_size, pooling_type='max'):
        super(SequentialAttention, self).__init__()
        self.hidden_size = hidden_size // 2
        self.encoder = nn.LSTM(input_size=hidden_size, hidden_size=self.hidden_size, bidirectional=True, num_layers=1)
        if pooling_type == 'mean':
            self.pooling = nn.AdaptiveAvgPool1d(output_size=1)
        else:
            self.pooling = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, *input):
        pass