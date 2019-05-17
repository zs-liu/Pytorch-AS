import torch
import torch.nn as nn
import torch.nn.functional as func
from model import Encoder
from model.simler import Simler


class Matcher(nn.Module):

    def __init__(self, batch_size, embedding_dim, hidden_dim, vocab_size, tagset_size=50, negative_size=10):
        super(Matcher, self).__init__()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.negative_size = negative_size
        self.encoder = Encoder(batch_size=batch_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                               bidirectional=True, vocab_size=vocab_size, dropout=0.5,
                               mlp_active=True, tagset_size=tagset_size)
        self.simler = Simler(negative_size=self.negative_size, threshold=1.5)

    def forward(self, q_batch, pa_batch, na_batch):
        q_out = self._single(q_batch[0], q_batch[1])
        pa_out = self._single(pa_batch[0], pa_batch[1])
        pre_na = self.pre_na(na_batch)
        na_out = self._single(pre_na[0], pre_na[1])
        loss, accu = self.simler(q_out, pa_out, na_out)
        return loss, accu

    def _single(self, s, s_length):
        s_length = s_length.long()
        if torch.cuda.is_available():
            s = s.cuda()
            s_length = s_length.cuda()
        s_length, s_pre_id = s_length.sort(0, descending=True)
        s = s[s_pre_id]
        s_out = self.encoder(s, s_length)
        s_inverse_id = torch.zeros(s_pre_id.size()[0]).long()
        if torch.cuda.is_available():
            s_inverse_id = s_inverse_id.cuda()
        for i in range(s_pre_id.size()[0]):
            s_inverse_id[s_pre_id[i]] = i

        q_out = s_out[s_inverse_id]
        return q_out

    def pre_na(self, na_batch):
        na_batch = na_batch[0]
        s_list = []
        length_list = []
        for j in range(0, self.batch_size):
            for i in range(0, self.negative_size):
                s_list.append(list(na_batch[i][0][j].numpy()))
                length_list.append(int(na_batch[i][1][j]))
        s_list = torch.LongTensor(s_list)
        length_list = torch.LongTensor(length_list)
        return s_list, length_list
