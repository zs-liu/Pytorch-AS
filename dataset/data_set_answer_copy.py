import torch
import numpy as np
from torch.utils.data import Dataset

from dataset import DataTaker


class InsuranceAnswerDataset(Dataset):

    def __init__(self, data_type='train', dataset_size=100, negative_size=10):
        self.data_list = []
        self.label_list = []
        self.dataset_size = dataset_size
        dt = DataTaker(dataset_size=self.dataset_size)
        if data_type == 'train':
            self.q_list = dt.read_train()
        elif data_type == 'test':
            self.q_list = dt.read_test()
        elif data_type == 'valid':
            self.q_list = dt.read_valid()
        self.a_list = dt.read_answer()
        self.max_length = max(dt.answer_max_len, dt.question_max_len)
        key_id = 0
        for line in self.q_list:
            for gt_id in line['ground_truth']:
                question = np.pad(line['question'], (0, self.max_length - len(line['question'])), 'constant',
                                  constant_values=0)
                answer = np.pad(self.a_list[gt_id], (0, self.max_length - len(self.a_list[gt_id])), 'constant',
                                constant_values=0)
                self.data_list.append([torch.LongTensor(question), torch.LongTensor(answer),
                                       len(line['question']), len(self.a_list[gt_id]), key_id])
                key_id += 1
                self.label_list.append(1)
            for nt_id in line['negative_pool'][:len(line['ground_truth'])]:
                question = np.pad(line['question'], (0, self.max_length - len(line['question'])), 'constant',
                                  constant_values=0)
                answer = np.pad(self.a_list[nt_id], (0, self.max_length - len(self.a_list[nt_id])), 'constant',
                                constant_values=0)
                self.data_list.append([torch.LongTensor(question), torch.LongTensor(answer),
                                       len(line['question']), len(self.a_list[nt_id]), key_id])
                key_id += 1
                self.label_list.append(0)

    def __getitem__(self, item):
        data = self.data_list[item]
        label = self.label_list[item]
        return data, label

    def __len__(self):
        return len(self.data_list)
