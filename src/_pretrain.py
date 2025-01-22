import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from _models.ae_model import AE
from _trainers.ae_trainer import AETrainer


class PreTrain:
    def __init__(self, device: torch.device, config: dict):
        self.device = device
        self.config = config
        self.dataset = config["dataset"]

    def embed(self, view: torch.Tensor, type_: str, view_idx: int) -> torch.Tensor:
        """
        Embeds the given view using pre-trained networks.

        Args:
            view (torch.Tensor):    The view to be embedded
            config (dict):          The configuration dictionary

        Returns:
            torch.Tensor:   The embedded view
        """
        if type_ == "vector":
            return self.embed_vector(view, view_idx)
        elif type_ == "text":
            return self.embed_text(view, view_idx)
        else:
            return

    def embed_vector(self, view: torch.Tensor, view_idx: int) -> torch.Tensor:
        """
        Embeds the given image view using AutoEncoders.

        Args:
            view (torch.Tensor):    The image view to be embedded
            config (dict):          The configuration dictionary

        Returns:
            torch.Tensor:   The embedded image view
        """

        architectures = self.config["ae"]["architectures"]
        architecture = architectures[view_idx]
        embedding_dim = architecture["output_dim"]
        if os.path.exists(f"weights/{self.dataset}{view_idx + 1}_ae_weights.pth"):
            self.ae = AE(
                architecture=None, input_dim=view.shape[1], embedding_dim=embedding_dim
            )
            self.ae.load_state_dict(
                torch.load(f"weights/{self.dataset}{view_idx + 1}_ae_weights.pth", map_location=self.device)
            )
            self.ae.to(self.device)

        else:
            trainer = AETrainer(self.config, self.device)
            self.ae = trainer.train(view, view_idx, architecture=architecture)

        self.ae.eval()
        with torch.no_grad():
            embedded = self.ae.encoder(view.to(self.device))
        return embedded

    def embed_text(self, view: torch.Tensor) -> torch.Tensor:
        """
        Embeds the given text view using BERT.

        Args:
            view (torch.Tensor):    The text view to be embedded
            config (dict):          The configuration dictionary

        Returns:
            torch.Tensor:   The embedded text view
        """
        return view
