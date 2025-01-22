import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from _models.ae_model import AE


class AETrainer:
    def __init__(self, config: dict, device: torch.device):
        self.dataset = config["dataset"]
        self.config = config["ae"]
        self.device = device

        self.lr = self.config["lr"]
        self.epochs = self.config["epochs"]
        self.lr_decay = self.config["lr_decay"]
        self.patience = self.config["patience"]
        self.n_samples = self.config["n_samples"]
        self.batch_size = self.config["batch_size"]

    def train(self, view: torch.Tensor, view_idx, architecture: dict) -> nn.Module:
        """
        Trains an AutoEncoder on the given view.

        Args:
            view (torch.Tensor):    The view to train the AutoEncoder on
            architecture (dict):    The architecture of the AutoEncoder

        Returns:
            nn.Module:  The trained AutoEncoder
        """
        embedding_dim = architecture["output_dim"]
        self.view = view.view(view.size(0), -1)
        self.criterion = nn.MSELoss()
        self.ae_net = AE(
            architecture, input_dim=self.view.shape[1], embedding_dim=embedding_dim
        ).to(self.device)
        self.optimizer = optim.Adam(self.ae_net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience
        )

        self.weights_path = f"./weights/{self.dataset}{view_idx + 1}_ae_weights.pth"

        train_loader, valid_loader = self._get_data_loader()

        print("Training Autoencoder:")
        for epoch in range(self.epochs):
            train_loss = 0.0
            for batch_x in train_loader:
                batch_x = batch_x.to(self.device)
                batch_x = batch_x.view(batch_x.size(0), -1)
                self.optimizer.zero_grad()
                output = self.ae_net(batch_x)
                loss = self.criterion(output, batch_x)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            valid_loss = self.validate(valid_loader)
            self.scheduler.step(valid_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            if current_lr <= self.config["min_lr"]:
                break
            print(
                "Epoch: {}/{}, Train Loss: {:.4f}, Valid Loss: {:.4f}, LR: {:.6f}".format(
                    epoch + 1, self.epochs, train_loss, valid_loss, current_lr
                )
            )

        torch.save(self.ae_net.state_dict(), self.weights_path)
        return self.ae_net

    def validate(self, valid_loader: DataLoader) -> float:
        """
        This function validates the autoencoder on the given data during the training process.

        Args:
            valid_loader (DataLoader):  the data to validate on

        Returns:
            float: the validation loss
        """

        self.ae_net.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_x in valid_loader:
                batch_x = batch_x.to(self.device)
                batch_x = batch_x.view(batch_x.size(0), -1)
                output = self.ae_net(batch_x)
                loss = self.criterion(output, batch_x)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        return valid_loss

    def _get_data_loader(self) -> tuple:
        """
        This function splits the data into train and validation sets
        and returns the corresponding data loaders.

        Returns:
            tuple: the train and validation data loaders
        """

        view = self.view[: self.n_samples]
        trainset_len = int(len(view) * 0.9)
        validset_len = len(view) - trainset_len
        trainset, validset = random_split(view, [trainset_len, validset_len])
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(validset, batch_size=self.batch_size, shuffle=False)
        return train_loader, valid_loader
