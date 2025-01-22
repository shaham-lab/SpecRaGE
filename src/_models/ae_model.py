import os
import torch
import torch.nn as nn
import torch.optim as optim


class AE(nn.Module):
    def __init__(self, architecture: dict, input_dim: int, embedding_dim: int):
        super(AE, self).__init__()
        self.architecture = architecture
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.embedding_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
