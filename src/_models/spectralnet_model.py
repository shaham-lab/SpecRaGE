import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


class SpectralNetModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(SpectralNetModel, self).__init__()
        self.architecture = architecture
        self.num_of_layers = self.architecture["n_layers"]
        self.layers = nn.ModuleList()
        self.input_dim = input_dim

        current_dim = self.input_dim
        for layer, dim in self.architecture.items():
            next_dim = dim
            if layer == "n_layers":
                continue
            if layer == "output_dim":
                layer = nn.Sequential(nn.Linear(current_dim, next_dim), nn.Tanh())
                self.layers.append(layer)
            else:
                layer = nn.Sequential(nn.Linear(current_dim, next_dim), nn.LeakyReLU())
                self.layers.append(layer)
                current_dim = next_dim

    def _make_orthonorm_weights(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Orthonormalize the output of the network using the Cholesky decomposition.

        Parameters
        ----------
        Y : torch.Tensor
            The output of the network.

        Returns
        -------
        torch.Tensor
            The orthonormalized output.

        Notes
        -----
        This function applies the Cholesky decomposition to orthonormalize the output (`Y`) of the network.
        The orthonormalized output is returned as a tensor.
        """

        m = Y.shape[0]

        _, R = torch.linalg.qr(Y)
        orthonorm_weights = np.sqrt(m) * torch.inverse(R)
        return orthonorm_weights

    def forward(self, x: torch.Tensor, is_orthonorm: bool = True) -> torch.Tensor:
        """
        This function performs the forward pass of the model.
        If is_orthonorm is True, the output of the network is orthonormalized
        using the Cholesky decomposition.

        Args:
            x (torch.Tensor):               The input tensor
            is_orthonorm (bool, optional):  Whether to orthonormalize the output or not.
                                            Defaults to True.

        Returns:
            torch.Tensor: The output tensor
        """

        for layer in self.layers:
            x = layer(x)

        return x
