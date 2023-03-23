# Chest X-ray alignment using Spatial Transformer Network

This is the `Python` implementation of the final project on Skoltech university MSc Data Science ML 2023 course "Spatial Transformer Network for Chest X-ray images preprocessing".

Team:

Andrey Galichin\
Evgeny Gurov\
Arkadiy Vladimirov

The repository contains reproducible `PyTorch` and `Pytorch Lightning` code for training and inference of our model which produces **properly aligned Chest X-ray images**. For convenience of performing different experiments effectively, we utilize `hydra` configs. We also provide some examples of our model performance, using Chest X-ray images from [Chest X-ray14](https://arxiv.org/pdf/1705.02315v5.pdf) dataset.

## Prerequisites

The training of our alignment model is GPU-based, inference can be done both on CPU or GPU. The only requirement for the gpu type is to be compatible with **CUDA 11.2**.

We highly recommend to use `conda` to build an appropriate environment. Working version could be created using `environment.yml` configuration file as follows:
```
conda env create -f environment.yml
```

## Repository structure

Evaluation of the model is issued in the form of pretty self-explanatory single jupyter notebook `Alignment (Evaluation).ipynb`. For convenience, some of the results will be presented further. Auxilary source code is moved to `.py` modules (`src/`), `hydra` configs are located in `configs/` to preserve the structure recommended by authors. 

### Training

Minimal required steps to run the training pipeline:

