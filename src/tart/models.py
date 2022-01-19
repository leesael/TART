from torch import nn
import torch

from tart.layers import Transition, BinaryDecision, MultiDecision, FullDenseDecision
from tart.predictors import EnsemblePredictor, DecisionPredictor


class TreeModel(nn.Module):
    def __init__(self, layers, leaf, style='ensemble'):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.leaf = leaf
        self.style = style
        self.num_leaves = leaf.in_nodes

        if style == 'ensemble':
            self.predictor = EnsemblePredictor()
        elif style == 'decision':
            self.predictor = DecisionPredictor()
        else:
            raise ValueError(style)

    def path(self, x):
        path = torch.ones((x.size(0), 1), device=x.device)
        for layer in self.layers:
            path = layer(x, path)
        return path

    def logit(self, x):
        return self.leaf(x).view(x.size(0), -1, self.num_leaves)

    def forward(self, x):
        return self.predictor(self.logit(x), self.path(x))

    def loss(self, x, y):
        return self.predictor.loss(self.logit(x), self.path(x), y)

    @torch.no_grad()
    def decision_path(self, x):
        mats = []
        paths = [torch.ones((x.size(0), 1), device=x.device)]
        for layer in self.layers:
            mats.append(layer.transition_matrix(x))
            paths.append(layer(x, paths[-1]))
        return mats, paths


def make_tree(num_features, num_classes, depth, window=2, leaf_layers=0,
              leaf_units=100, style='decision', temperature=1, decision_units=0,
              activation='softmax', decision_shared: bool = False):
    layers = []
    for d in range(depth):
        num_nodes = int(window ** d)
        if window == 2:
            layer = BinaryDecision(num_features, num_nodes,
                                   num_units=decision_units,
                                   temperature=temperature,
                                   shared=decision_shared)
        elif window > 2:
            layer = MultiDecision(num_features, num_nodes,
                                  window=window,
                                  activation=activation,
                                  hidden_units=decision_units,
                                  temperature=temperature,
                                  shared=decision_shared)
        else:
            raise ValueError()
        layers.append(layer)
    num_leaves = int(window ** depth)
    leaf = Transition(
        num_features, num_leaves, num_classes, leaf_layers, leaf_units)
    return TreeModel(layers, leaf, style)


def make_network(num_features, num_classes, depth, style='decision',
                 num_nodes=16, activation='softmax'):
    layers = []
    for d in range(depth):
        if d == 0:
            layer = FullDenseDecision(num_features, 1, num_nodes, activation)
        else:
            layer = FullDenseDecision(
                num_features, num_nodes, num_nodes, activation)
        layers.append(layer)
    leaf = Transition(num_features, num_nodes, num_classes)
    return TreeModel(layers, leaf, style)
