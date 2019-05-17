import torch
import torch.nn as nn
import torch.nn.functional as func

from model.attentioner import SelfAttention


class Encoder(nn.Module):

    def __init__(self, batch_size, embedding_dim, hidden_dim, vocab_size, num_layers=1, dropout=0, bidirectional=False,
                 pooling_type='max', mlp_active=True, tagset_size=50, self_attention=False):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=1)
        if self.num_layers == 1:
            dropout = 0
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            dropout=dropout, bidirectional=bidirectional)

        self.pooling_type = pooling_type
        if pooling_type == 'mean':
            self.pooling = nn.AdaptiveAvgPool1d(output_size=1)
        elif pooling_type == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(output_size=1)
        else:
            self.pooling = nn.Identity()

        if self_attention:
            if self.bidirectional:
                self.attention = SelfAttention(input_dim=self.num_layers * self.hidden_dim * 2,
                                               hidden_dim=self.hidden_dim)
            else:
                self.attention = SelfAttention(input_dim=self.num_layers * self.hidden_dim,
                                               hidden_dim=self.hidden_dim // 2)
        else:
            self.attention = nn.Identity()

        self.mlp_active = mlp_active

        if self.mlp_active:
            self.dropout = nn.Dropout()
            if bidirectional:
                self.hidden2tag1 = nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim // 2)
                self.hidden2tag2 = nn.Linear(in_features=hidden_dim // 2, out_features=tagset_size)
            else:
                self.hidden2tag1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim // 4)
                self.hidden2tag2 = nn.Linear(in_features=hidden_dim // 4, out_features=tagset_size)

        self.hidden = self._init_hidden(self.batch_size)

    def forward(self, s, s_length):
        embed = self.embedding(s)
        x = embed.view(-1, embed.size(1), self.embedding_dim)
        self.hidden = self._init_hidden(x.shape[0])
        x = nn.utils.rnn.pack_padded_sequence(x, s_length, batch_first=True)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = torch.transpose(lstm_out, 1, 2)  # batch size, hidden dim (* 2 if bid), len
        lstm_out = self.attention(lstm_out)
        if self.pooling_type == 'raw':
            return lstm_out
        tag_vector = self.pooling(lstm_out).squeeze(2)  # batch size, hidden dim (* 2 if bid)
        if self.mlp_active:
            tag_vector = self.hidden2tag1(func.relu(tag_vector))
            tag_vector = self.hidden2tag2(self.dropout(func.relu(tag_vector)))
        return tag_vector

    def _init_hidden(self, h0):
        if self.bidirectional:
            hidden_matrix_1 = torch.zeros(2 * self.num_layers, h0, self.hidden_dim)
            hidden_matrix_2 = torch.zeros(2 * self.num_layers, h0, self.hidden_dim)
        else:
            hidden_matrix_1 = torch.zeros(self.num_layers, h0, self.hidden_dim)
            hidden_matrix_2 = torch.zeros(self.num_layers, h0, self.hidden_dim)
        if torch.cuda.is_available():
            hidden_matrix_1 = hidden_matrix_1.cuda()
            hidden_matrix_2 = hidden_matrix_2.cuda()
        return hidden_matrix_1, hidden_matrix_2
