from torch import nn


class MLP(nn.Module):
    def __init__(self, num_features, num_classes, num_units=100, num_layers=10, drop_prob=0.15):
        super().__init__()
        layers = []
        for n in range(num_layers - 1):
            layers.extend([nn.Linear(num_features, num_units),
                           nn.ELU(),
                           nn.Dropout(drop_prob)])
            num_features = num_units
        layers.append(nn.Linear(num_features, num_classes))
        self.layers = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x)

    def loss(self, x, y):
        return self.loss_func(self.forward(x), y)

    def describe(self, x):
        out = x
        out_list = []
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i % 3 == 0:
                out_list.append(out)
        return out_list
