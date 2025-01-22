import torch
import numpy as np

from _pretrain import *
from _trainers.siamese_trainer import *
from _trainers.multispecnet_trainer import *

from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class SpecRaGE:
    def __init__(self, n_clusters: int, config: dict):
        """
        Args:
            n_clusters (int):   The dimension of the projection subspace
            config (dict):      The configuration dictionary
        """

        self.n_clusters = n_clusters
        self.config = config
        self.embeddings_ = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, views: list, labels: torch.Tensor = None):
        """
        Performs the main training loop for the SpectralNet model.

        Args:
            views (list):       Multi-view data to train the networks on. Each element in the list is a
                                different view with shape nxd (n samples, d features).
            y (torch.Tensor):   Labels in case there are any. Defaults to None.
        """

        dataset = self.config["dataset"]
        should_use_ae = self.config["should_use_ae"]
        should_use_siamese = self.config["should_use_siamese"]
        create_weights_dir()

        siamese_nets = []
        embedded_views = []
        types = self.config["datatypes"]
        pre_train = PreTrain(self.device, self.config)

        if should_use_ae:
            for i, (view, view_type) in enumerate(zip(views, types)):
                view = pre_train.embed(view, view_type, i)
                embedded_views.append(view)
            views = embedded_views


        if should_use_siamese:
            for i, view in enumerate(views):
                weights_path = f"weights/{dataset}{i + 1}_siamese_weights.pth"
                architectures = self.config["siamese"]["architectures"]
                if os.path.exists(weights_path):
                    siamese_net = SiameseNet(
                        architectures[i], input_dim=view.shape[1]
                    ).to(self.device)
                    siamese_net.load_state_dict(torch.load(weights_path, map_location=self.device))
                else:
                    siamese_trainer = SiameseTrainer(self.config, self.device)
                    siamese_net = siamese_trainer.train(
                        view, architecture=architectures[i]
                    )
                    torch.save(siamese_net.state_dict(), weights_path)

                siamese_nets.append(siamese_net)

        specrage_trainer = SpecRaGETrainer(self.config, self.device)
        self.specrage_model = specrage_trainer.train(
            views, labels, siamese_nets
        )

    def predict(self, views: list) -> np.ndarray:
        """
        Predicts the cluster assignments for the given data.

        Args:
            views (list):   Data to be clustered

        Returns:
            np.ndarray:  the cluster assignments for the given data

        """

        types = self.config["datatypes"]
        should_use_ae = self.config["should_use_ae"]
        pre_train = PreTrain(self.device, self.config)
        embedded_views = []

        if should_use_ae:
            for i, (view, view_type) in enumerate(zip(views, types)):
                view = pre_train.embed(view, view_type, i)
                embedded_views.append(view)
            views = embedded_views

        else:
            for i in range(len(views)):
                views[i] = views[i].to(self.device)

        with torch.no_grad():
            embeddings = (
                self.specrage_model(views, is_orthonorm=False)[0]
                .detach()
                .cpu()
                .numpy()
            )
            self.embeddings_ = embeddings

        return self.embeddings_

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Performs k-means clustering on the spectral-embedding space.

        Args:
            embeddings (np.ndarray):   the spectral-embedding space

        Returns:
            np.ndarray:  the cluster assignments for the given data
        """
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=100).fit(embeddings)
        cluster_assignments = kmeans.predict(embeddings)
        return cluster_assignments
    
    
    def classify(self, embeddings: np.ndarray, labels) -> np.ndarray:
        """
        Performs k-means clustering on the spectral-embedding space.

        Args:
            embeddings (np.ndarray):   the spectral-embedding space

        Returns:
            np.ndarray:  the cluster assignments for the given data
        """
        # embeddings = embeddings.detach().cpu().numpy()
        self._train_svm(embeddings, labels)
        predictions = self._svm_predict(embeddings)
        return predictions

    def _train_svm(self, embeddings: np.ndarray, labels):
        # self.svm = make_pipeline(StandardScaler(), SVC(gamma='scale'))
        self.svm = make_pipeline(StandardScaler(), LinearSVC(dual=False))
        self.svm.fit(embeddings, labels)

    def _svm_predict(self, embeddings: np.ndarray):
        predictions = self.svm.predict(embeddings)
        return predictions
