import torch
import numpy as np
from torch.utils.data import Dataset

from dataset import DataTaker


class InsuranceQuestionDataset(Dataset):

    def __init__(self, data_type='train', dataset_size=100):
        self.data_list = []
        self.dataset_size = dataset_size
        dt = DataTaker(dataset_size=self.dataset_size)
        if data_type == 'train':
            self.q_list = dt.read_train()
        elif data_type == 'test':
            self.q_list = dt.read_test()
        elif data_type == 'valid':
            self.q_list = dt.read_valid()
        self.max_length = dt.question_max_len
        key_id = 0
        for line in self.q_list:
            for gt_id in line['ground_truth']:
                question = np.pad(line['question'], (0, self.max_length - len(line['question'])), 'constant',
                                  constant_values=0)
                self.data_list.append([question, len(line['question']), key_id])
                key_id += 1

    def __getitem__(self, item):
        data = self.data_list[item]
        label = None
        return data, label

    def __len__(self):
        return len(self.data_list)
