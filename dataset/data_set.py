import torch
import numpy as np
from torch.utils.data import Dataset

from dataset import DataTaker


class InsuranceAnswerDataset(Dataset):

    def __init__(self, data_type='train', dataset_size=100, negative_size=10, full_flag=False, full_size=0):
        """
        :param data_type: 'train', 'valid' or 'test'
        :param dataset_size: relate to the dataset size, but not equivalent
        :param negative_size: negative answer number for one positive answer
        """
        self.data_list = []
        self.dataset_size = dataset_size
        self.negative_size = negative_size
        if full_flag:
            self.negative_size = full_size - 10
        dt = DataTaker(dataset_size=self.dataset_size, full_flag=full_flag, full_size=full_size)
        if data_type == 'train':
            self.q_list = dt.read_train()
        elif data_type == 'test':
            self.q_list = dt.read_test()
        elif data_type == 'valid':
            self.q_list = dt.read_valid()
        self.a_list = dt.read_answer()
        self.max_q_length = dt.question_max_len
        self.max_a_length = dt.answer_max_len
        key_id = 0
        for line in self.q_list:
            for gt_id in line['ground_truth']:
                temp_data = []
                question = np.pad(line['question'], (0, self.max_q_length - len(line['question'])), 'constant',
                                  constant_values=0)
                temp_data.append([question, len(line['question']), key_id])
                answer = np.pad(self.a_list[gt_id], (0, self.max_a_length - len(self.a_list[gt_id])), 'constant',
                                constant_values=0)
                temp_data.append([torch.LongTensor(answer), len(self.a_list[gt_id]), key_id])
                nt_id_list = line['negative_pool']
                np.random.shuffle(nt_id_list)
                nt_list = []
                for nt_id in nt_id_list[:self.negative_size]:
                    answer = np.pad(self.a_list[nt_id], (0, self.max_a_length - len(self.a_list[nt_id])), 'constant',
                                    constant_values=0)
                    nt_list.append([torch.LongTensor(answer), len(self.a_list[nt_id]), key_id])
                key_id += 1
                temp_data.append(nt_list)
                self.data_list.append(temp_data)

    def __getitem__(self, item):
        data = self.data_list[item]
        return data, 1

    def __len__(self):
        return len(self.data_list)
