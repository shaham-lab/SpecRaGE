# MvSpecNet
This is the official PyTorch implementation of MvSpecNet from the paper "Scalable and Generalizable Multi-view Deep Spectral Representation Learning".

<p align="center">
    <img src="https://github.com/shaham-lab/SpecRaGE/blob/main/figures/MvSpecNet.png">

## Installation
To run the project, clone this repo and then create a conda environment via:

```bash
conda env create -f environment.yml
```
Subsequently, activate this environment:

```bash
conda activate specrage
```

## Clustering 
To run a clustering experiment for the BDGP dataset, for example, cd to the src directory and run:

```bash
python3 cluster.py bdgp
```

The output should look like the following:
<p align="center">
    <img src="https://github.com/shaham-lab/SpecRaGE/blob/main/figures/bdgp_cluster_loss.png">

## Classification 
To run a classification experiment, cd to the src directory and run:
```bash
python3 classify.py bdgp
```
In this case, the output should look like the following:
<p align="center">
    <img src="https://github.com/shaham-lab/SpecRaGE/blob/main/figures/bdgp_classify_loss.png">

## Visualization of the unified representation for the BDGP dataset
<p align="center">
    <img src="https://github.com/shaham-lab/SpecRaGE/blob/main/figures/representation_visu.png">



