import sys
import json
import torch
import random
import numpy as np
import wandb

from sklearn.metrics import precision_score, f1_score

from _utils import *
from _data import load_data
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
    labels = labels_test
    labels = labels.detach().cpu().numpy()
    
    embeddings = specrage.predict(Views_test)
    predictions = specrage.classify(embeddings, labels)

    acc_score = np.mean(predictions == labels)
    fsc_score = f1_score(labels, predictions, average='weighted')
    pre_score = precision_score(labels, predictions, average='weighted')

    print(f"Accuracy:  {np.round(acc_score, 4)}")
    print(f"F-score:  {np.round(fsc_score, 4)}")
    print(f"Precision:  {np.round(pre_score, 4)}")



if __name__ == "__main__":
    main()
