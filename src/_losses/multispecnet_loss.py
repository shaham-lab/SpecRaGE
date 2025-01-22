import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from .spectralnet_loss import SpectralNetLoss


class SpecRaGELoss(nn.Module):
    def __init__(self):
        super(SpecRaGELoss, self).__init__()

    def forward(
        self, Ws: list, Y: torch.Tensor, is_normalized: bool = False,
    ) -> torch.Tensor:
        """
        This function computes the loss of the MultiSpectralNet model.
        The loss is the sum of the rayleigh quotient of the Laplacian matrix obtained from each W,
        and the orthonormalized output of the network.

        Args:
            Ws (list):                             Affinity matrices
            Y (torch.Tensor):                      Outputs of the network
            is_normalized (bool, optional):        Whether to use the normalized Laplacian matrix or not.

        Returns:
            torch.Tensor: The loss
        """
        num_of_views = len(Ws)
        m = Y.shape[0]
        loss = 0
        for i in range(num_of_views):
            loss += SpectralNetLoss()(Ws[i], Y, is_normalized)

        return loss / (m * num_of_views)


