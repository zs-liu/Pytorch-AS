import numpy as np


class DataTaker:

    def __init__(self, data_path='/cos_person/data/insuranceQA/V2/', dataset_type='token', dataset_size=100,
                 full_flag=False, full_size=0):
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size
        self.vocabulary_path = self.data_path + 'vocabulary'
        self.answer_path = self.data_path + 'InsuranceQA.label2answer.' + self.dataset_type + '.encoded'
        self.answer_max_len = 0
        self.train_path = self.data_path + 'InsuranceQA.question.anslabel.' + self.dataset_type + '.' \
                          + str(self.dataset_size) + '.pool.solr.train.encoded'
        self.valid_path = self.data_path + 'InsuranceQA.question.anslabel.' + self.dataset_type + '.' \
                          + str(self.dataset_size) + '.pool.solr.valid.encoded'
        self.test_path = self.data_path + 'InsuranceQA.question.anslabel.' + self.dataset_type + '.' \
                         + str(self.dataset_size) + '.pool.solr.test.encoded'
        self.question_max_len = 0
        self.full_flag = full_flag
        self.full_size = full_size
        if self.full_flag and self.full_size == 0:
            self.full_size = 1500

    def read_vocabulary(self):
        path = self.vocabulary_path
        f = open(path, 'rb')
        vocabulary = {}
        r_vocabulary = {}
        for line in f.readlines():
            line = line.decode()
            idx, word = line.strip().split('\t')
            idx = int(idx[4:])
            vocabulary[idx] = word.lower()
            if r_vocabulary.get(word.lower()) is None:
                r_vocabulary[word.lower()] = idx
        f.close()
        return vocabulary, r_vocabulary

    def read_train(self):
        path = self.train_path
        return self._read_question(path)

    def read_valid(self):
        path = self.valid_path
        return self._read_question(path)

    def read_test(self):
        path = self.test_path
        return self._read_question(path)

    def _read_question(self, path):
        data = []
        f = open(path, 'rb')
        for line in f.readlines():
            line = line.decode()
            domain, question, p_answer, n_answer = line.strip().split('\t')
            p_answer = p_answer.split(' ')
            p_answer = [int(x) for x in p_answer]
            if not self.full_flag:
                n_answer = n_answer.split(' ')
                n_answer = [int(x) for x in n_answer]
            else:
                temp = np.arange(self.full_size) + 1
                np.random.shuffle(temp)
                n_answer = list(temp)
            question = question.split(' ')
            question = [int(x[4:]) for x in question]
            for gt in p_answer:
                try:
                    n_answer.remove(gt)
                except ValueError:
                    pass
            line = {'domain': domain, 'question': question, 'ground_truth': p_answer, 'negative_pool': n_answer}
            self.question_max_len = max(self.question_max_len, len(question))
            data.append(line)
        f.close()
        return data

    def read_answer(self):
        path = self.answer_path
        answer = {}
        f = open(path, 'rb')
        for line in f.readlines():
            line = line.decode()
            idx, _answer = line.strip().split('\t')
            idx = int(idx)
            _answer = _answer.split(' ')
            _answer = [int(x[4:]) for x in _answer]
            answer[idx] = _answer
            self.answer_max_len = max(self.answer_max_len, len(_answer))
        f.close()
        return answer

    def test(self):
        try:
            f = open(self.answer_path, 'rb')
            f.close()
            f = open(self.vocabulary_path)
            f.close()
            f = open(self.train_path, 'rb')
            f.close()
            f = open(self.test_path, 'rb')
            f.close()
            f = open(self.valid_path, 'rb')
            f.close()
        except FileNotFoundError:
            print('Data File Not Found.')
            return False
        return True
