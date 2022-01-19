from torch import nn
import torch


class EnsemblePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logit, path):
        return torch.bmm(torch.softmax(logit, dim=1), path.unsqueeze(2)).squeeze(2)

    def loss(self, logit, path, y):
        y = y.unsqueeze(1).expand(y.size(0), path.size(1))
        return (path * self.loss_func(logit, y)).sum(dim=1).mean()


class DecisionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logit, path):
        return torch.softmax(logit[torch.arange(logit.size(0)), :, path.argmax(dim=1)], dim=1)

    def loss(self, logit, path, y):
        y = y.unsqueeze(1).expand(y.size(0), path.size(1))
        return (path * self.loss_func(logit, y)).sum(dim=1).mean()
