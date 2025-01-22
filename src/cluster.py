import sys
import json
import torch
import random
import numpy as np
import wandb
import time

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from _utils import *
from _data import load_data
from _metrics import Metrics
from _model import SpecRaGE



def set_seed(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    dataset = sys.argv[1]

    config_path = f"config/{dataset}.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    dataset = config["dataset"]
    n_clusters = config["n_clusters"]

    Views_train, Views_test, labels_train, labels_test = load_data(dataset)
    

    specrage = SpecRaGE(n_clusters=n_clusters, config=config)
    specrage.fit(Views_train, labels_train)

    embeddings = specrage.predict(Views_test)
    cluster_assignments = specrage.cluster(embeddings)
    labels = labels_test
    labels = labels.detach().cpu().numpy()
    

    if labels is not None:
        acc_score = Metrics.acc_score(cluster_assignments, labels, n_clusters)
        nmi_score = Metrics.nmi_score(cluster_assignments, labels)
        ari_score = Metrics.ari_score(cluster_assignments, labels)

        print(f"ACC: {np.round(acc_score, 3)}")
        print(f"NMI: {np.round(nmi_score, 3)}")
        print(f"ARI: {np.round(ari_score, 3)}")


if __name__ == "__main__":
   main()
 
