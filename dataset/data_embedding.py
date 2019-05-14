import numpy as np
from dataset import DataTaker


class DataEmbedding:

    def __init__(self, embedding_path='/cos_person/data/embedding/glove.6B.50d.txt', embedding_dim=50, embedding_size=70000):
        """
        :param embedding_path: the original embedding data
        :param embedding_dim: embedding dim, corresponding to the data
        :param embedding_size: embedding size, corresponding size
        """
        self.embedding_path = embedding_path
        self.embedding_dim = embedding_dim
        self.embedding_size = embedding_size

    def get_embedding_matrix(self):
        """
        produce the embedding matrix for data
        :return: embedding matrix, to be used as the initial weight of embedding layer
        """
        path = self.embedding_path
        dt = DataTaker()
        _, r_dict = dt.read_vocabulary()
        matrix = np.zeros((self.embedding_size, self.embedding_dim))
        f = open(path, 'rb')
        for line in f.readlines():
            line = line.decode()
            line = line.split(' ')
            word = line[0]
            idx = r_dict.get(word, -1)
            if idx != -1:
                vector = [float(x) for x in line[1:]]
                matrix[idx][:] = vector
        return matrix
