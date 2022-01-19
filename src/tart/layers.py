import math

import torch
from torch import nn
from torch.nn import init

from .activation import to_activation


def to_conv_mask(num_nodes, window, stride, padding):
    mask = torch.ones(num_nodes, window, dtype=torch.bool)
    for i in range(num_nodes):
        for j in range(window):
            dist_front = i * stride + j
            dist_end = (num_nodes - i - 1) * stride + (window - j - 1)
            if dist_front < padding or dist_end < padding:
                mask[i, j] = 0
    return mask.t().view(1, window, num_nodes)


class Bias(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        bound = 1 / math.sqrt(in_features)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.bias.unsqueeze(0).expand(x.size(0), -1)


class GroupedLinear(nn.Module):
    def __init__(self, num_features, in_nodes, out_nodes, reshape=True):
        super().__init__()
        self.layer = nn.Conv1d(in_channels=in_nodes * num_features,
                               out_channels=in_nodes * out_nodes,
                               kernel_size=1,
                               groups=in_nodes)
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.reshape = reshape

    def forward(self, x):
        out = self.layer(x.unsqueeze(2))
        if self.reshape:
            out = out.view(x.size(0), self.in_nodes, self.out_nodes).transpose(1, 2)
        else:
            out = out.view(x.size(0), -1)
        return out


class Block(nn.Module):
    def __init__(self, layer, dropout=0.15):
        super().__init__()
        self.layers = nn.Sequential(layer, nn.ELU(), nn.Dropout(p=dropout))

    def forward(self, x):
        return self.layers(x)


class Transition(nn.Module):
    def __init__(self, num_features, in_nodes, out_nodes, num_layers=1,
                 hidden_units=0, shared=False):
        super().__init__()
        if num_layers == 0:
            layers = [Bias(num_features, out_nodes * in_nodes)]
        elif num_layers == 1:
            layers = [nn.Linear(num_features, out_nodes * in_nodes)]
        elif shared:  # Share the hidden units across all decision units.
            layers = [Block(nn.Linear(num_features, hidden_units))]
            for i in range(num_layers - 2):
                layers.append(Block(nn.Linear(hidden_units, hidden_units)))
            layers.append(nn.Linear(hidden_units, out_nodes * in_nodes))
        else:
            layers = [Block(nn.Linear(num_features, hidden_units * in_nodes))]
            for i in range(num_layers - 2):
                layers.append(Block(GroupedLinear(
                    hidden_units, in_nodes, hidden_units, reshape=False)))
            layers.append(GroupedLinear(hidden_units, in_nodes, out_nodes))

        self.num_layers = num_layers
        self.layers = nn.Sequential(*layers)
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes

    def forward(self, x):
        return self.layers(x).view(x.size(0), self.out_nodes, self.in_nodes)


class BinaryDecision(nn.Module):
    def __init__(self, num_features, num_nodes, num_units=0, temperature=1,
                 shared=False):
        super().__init__()
        out_nodes = 1
        num_layers = 2 if num_units > 0 else 1
        self.num_nodes = num_nodes
        self.transition = Transition(num_features, num_nodes, out_nodes,
                                     num_layers, num_units, shared)
        self.temperature = temperature

    def forward(self, x, path):
        assert x.size(0) == path.size(0) and path.size(1) == self.num_nodes
        prob_right = torch.sigmoid(self.transition(x).squeeze(1) / self.temperature)
        new_prob = torch.stack((1 - prob_right, prob_right), dim=2)
        return (path.unsqueeze(2) * new_prob).view(x.size(0), -1)

    def transition_vector(self, x):
        prob_right = torch.sigmoid(self.transition(x).squeeze(1) / self.temperature)
        return torch.stack((1 - prob_right, prob_right), dim=2).view(x.size(0), -1)

    def transition_matrix(self, x):
        path = torch.ones(x.size(0), self.num_nodes, device=x.device)
        out_vec = self.forward(x, path)
        out_mat = torch.zeros(x.size(0), 2 * self.num_nodes, self.num_nodes)
        for i in range(self.num_nodes):
            out_mat[:, 2 * i:2 * (i + 1), i] = out_vec[:, 2 * i:2 * (i + 1)]
        return out_mat


class MultiDecision(nn.Module):
    def __init__(self, num_features, num_nodes, window, stride=None, padding=0,
                 activation='softmax', hidden_units=0, temperature=1, shared=False):
        super().__init__()
        stride = window if stride is None else stride
        self.num_nodes = num_nodes
        self.window = window
        self.stride = stride
        self.padding = padding
        self.temperature = temperature

        self.mask = nn.Parameter(to_conv_mask(num_nodes, window, stride, padding),
                                 requires_grad=False)
        num_layers = 2 if hidden_units > 0 else 1
        self.transform = Transition(num_features, num_nodes, window, num_layers,
                                    hidden_units, shared)
        self.activation = to_activation(activation, dim=1)

        self.conv = nn.ConvTranspose1d(window, 1, window, stride, padding, bias=False)
        self.conv.weight.requires_grad = False
        self.conv.weight.fill_(0)
        self.conv.weight.squeeze(1).fill_diagonal_(1)

    def out_nodes(self):
        return self.window + (self.num_nodes - 1) * self.stride - 2 * self.padding

    def transition_block(self, x):
        return self.activation(self.transform(x) / self.temperature, self.mask)

    def forward(self, x, path):
        return self.conv(self.transition_block(x) * path.unsqueeze(1)).squeeze(1)

    def transition_matrix(self, x):
        out = self.transition_block(x)
        out_list = []
        path = torch.zeros((1, 1, self.num_nodes), device=x.device)
        for i in range(self.num_nodes):
            path.fill_(0)
            path[0, 0, i] = 1
            out_list.append(self.conv(out * path).squeeze(1))
        return torch.stack(out_list, dim=2)


class FullDenseDecision(nn.Module):
    def __init__(self, num_features, in_nodes, out_nodes, activation='softmax'):
        super().__init__()
        self.layers = Transition(num_features, in_nodes, out_nodes)
        self.activation = to_activation(activation, dim=1)

    def forward(self, x, path):
        mat_t = self.activation(self.layers(x))
        return torch.bmm(mat_t, path.unsqueeze(2)).squeeze(2)
