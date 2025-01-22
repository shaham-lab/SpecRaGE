# SpecRaGE
This is the official PyTorch implementation of SpecRaGE from the paper "Generalizable and Robust Spectral Method for Multi-view Representation Learning".

<p align="center">
    <img src="https://github.com/shaham-lab/SpecRaGE/blob/main/figures/SpecRaGE.png" width="600">
</p>



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



