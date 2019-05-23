import torch
import torch.nn as nn
import torch.nn.functional as F


class SequentialAttention(nn.Module):
    def __init__(self, hidden_size, pooling_type='mean'):
        super(SequentialAttention, self).__init__()
        self.hidden_size = hidden_size // 2
        self.encoder = nn.LSTM(input_size=hidden_size, hidden_size=self.hidden_size, bidirectional=True, num_layers=1)
        if pooling_type == 'mean':
            self.pooling = torch.mean
        elif pooling_type == 'max':
            self.pooling = torch.max
        else:
            self.pooling = nn.Identity()

    def forward(self, from_q, to_a):
        temp = torch.mul(from_q, to_a)  # batch_size, len, hidden_dim
        temp = temp.permute(1, 0, 2)  # len, batch_size, hidden_dim
        hidden = self._init_hidden(temp.shape[1])
        lstm_out, _ = self.encoder(temp, hidden)
        lstm_out = lstm_out.permute(1, 0, 2)  # batch_size, len, hidden_dim
        weight = self.pooling(lstm_out).unsqueeze(2)  # batch_size, len, 1
        weight = F.softmax(weight, dim=1)  # batch_size, len, 1
        return torch.sum(to_a * weight, 1)

    def _init_hidden(self, h0):
        hidden_matrix_1 = torch.zeros(2 * self.num_layers, h0, self.hidden_dim)
        hidden_matrix_2 = torch.zeros(2 * self.num_layers, h0, self.hidden_dim)
        if torch.cuda.is_available():
            hidden_matrix_1 = hidden_matrix_1.cuda()
            hidden_matrix_2 = hidden_matrix_2.cuda()
        return hidden_matrix_1, hidden_matrix_2


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.weight = torch.zeros(hidden_dim)
        if torch.cuda.is_available():
            self.weight = self.weight.cuda()
        self._reset_para()

    def forward(self, vector):
        vector = vector.permute((0, 2, 1))  # batch size, len, input him
        alpha = self.linear(vector)  # batch size, len, hidden him
        alpha = alpha * self.weight  # batch size, len, hidden dim
        alpha = torch.sum(alpha, dim=2, keepdim=True)  # batch size, len, 1
        alpha = F.softmax(alpha, dim=1)  # batch size, len, 1
        vector = vector * alpha  # batch size, len, input him
        vector = vector.permute((0, 2, 1))   # batch size, input him, len
        return vector

    def _reset_para(self):
        stdv = 0.1
        self.weight.data.uniform_(-stdv, stdv)
