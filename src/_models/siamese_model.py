import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


class SiameseNet(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(SiameseNet, self).__init__()
        self.architecture = architecture
        self.num_of_layers = self.architecture["n_layers"]
        self.layers = nn.ModuleList()

        current_dim = input_dim
        for layer, dim in self.architecture.items():
            if layer == "n_layers":
                continue
            next_dim = dim
            layer = nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU())
            self.layers.append(layer)
            current_dim = next_dim

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple:
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2
