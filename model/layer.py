import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dimensions, input_dim=768, activation='relu', dropout=0.):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.dimensions = dimensions
        self.activation = activation
        self.dropout = dropout

        # MLP architecture
        self.linears = nn.ModuleList([nn.Linear(input_dim, dimensions[0])])
        for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:]):
            self.linears.append(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        num_layers = len(self.linears)
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if (i < (num_layers - 1)):
                x = F.__dict__[self.activation](x)

            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
