import torch
import torch.nn as nn


class CosSim(nn.Module):
    def __init__(self):
        super(CosSim, self).__init__()
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, vector_a, vector_b):
        return self.similarity(vector_a, vector_b)


class AesdSim(nn.Module):

    def __init__(self, gamma=1.0, c=1.0):
        super(AesdSim, self).__init__()
        self.gamma = gamma
        self.c = c
        self.distance = nn.PairwiseDistance(p=2., eps=1e-6)

    def forward(self, vector_a, vector_b):
        sim_1 = torch.reciprocal(1 + self.distance(vector_a, vector_b))
        sim_2 = torch.reciprocal(1 + torch.exp(-self.gamma * (torch.mul(vector_a, vector_b).sum() + self.c)))
        sim = 0.5 * torch.add(sim_1, sim_2)
        if torch.cuda.is_available():
            sim = sim.cuda()
        return sim


class GesdSim(nn.Module):

    def __init__(self, gamma=1.0, c=1.0):
        super(GesdSim, self).__init__()
        self.gamma = gamma
        self.c = c
        self.distance = nn.PairwiseDistance(p=2., eps=1e-6)

    def forward(self, vector_a, vector_b):
        sim_1 = torch.reciprocal(1 + self.distance(vector_a, vector_b))
        sim_2 = torch.reciprocal(1 + torch.exp(-self.gamma * (torch.mul(vector_a, vector_b).sum() + self.c)))
        sim = torch.mul(sim_1, sim_2)
        if torch.cuda.is_available():
            sim = sim.cuda()
        return sim


class HyperbolicSim(nn.Module):

    def __init__(self, weight, bias):
        super(HyperbolicSim, self).__init__()
        self.weight = weight
        self.bias = bias
        self.distance = nn.PairwiseDistance(p=2., eps=1e-6)

    def forward(self, vector_a, vector_b):
        ed = torch.pow(self.distance(vector_a, vector_b), exponent=2)
        ea = torch.pow(self.distance(vector_a, vector_a), exponent=2)
        eb = torch.pow(self.distance(vector_b, vector_b), exponent=2)
        temp = 1 + 2 * ed * torch.reciprocal(1 - ea) * torch.reciprocal(1 - eb)
        d = torch.log(temp + torch.sqrt(torch.pow(temp, 2) - 1))
