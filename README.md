# ✨ 🐴 🔥 PONITA-JAX

This is a Jax implementation of the original [Ponita codebase](https://github.com/ebekkers/ponita) corresponding to the paper!

ACCEPTED AT [ICLR 2024](https://openreview.net/forum?id=dPHLbUqGbr)!

## What is this repository about?
This repository contains the code for the paper [Fast, Expressive SE(n) Equivariant Networks through Weight-Sharing in Position-Orientation Space](https://arxiv.org/abs/2310.02970). We propose **PONITA**: a simple fully convolutional SE(n) equivariant architecture. We developed it primarily for 3D point-cloud data, but the method is also applicable to 2D point clouds and 2D/3D images/volumes (though not yet with this repo). PONITA is an equivariant model that does not require working with steerable/Clebsch-Gordan methods, but has the same capabilities in that __it can handle scalars and vectors__ equally well. Moreover, since it does not depend on Clebsch-Gordan tensor products __PONITA is much faster__ than the typical steerable/tensor field network!

See below results and code for benchmarks for 2D (**super-pixel MNIST**) and 3D point clouds with vector attributes (**n-body**) and without (**MD17**), as well as an example of position-orientation space point clouds (**QM9**)! Results for equivariant generative modeling are in the paper (which will soon be updated with the MNIST and QM9 regression results as presented below).

## About the name
PONITA is an acronym for Position-Orientation space Networks based on InvarianT Attributes. We believe this acronym is apt for the method for two reasons. Firstly, PONITA sounds like "bonita" ✨ which means pretty in Spanish, we personally think the architecture is pretty and elegant. Secondly, [Ponyta](https://bulbapedia.bulbagarden.net/wiki/Ponyta_(Pok%C3%A9mon)) 🐴 🔥 is a fire Pokémon which is known to be very fast, our method is fast as well.

## Conda environment
In order to run the code in this repository install the following conda environment
```
conda create --yes --name ponita-jax python=3.10 numpy matplotlib
conda activate ponita-jax
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip3 install wandb
pip3 install hydra-core
pip3 install rdkit
```

## Acknowledgements
The experimental setup builds upon the code bases of [EGNN repository](https://github.com/vgsatorras/egnn) and [EDM repository](https://github.com/ehoogeboom/e3_diffusion_for_molecules). The grid construction code is adapted from [Regular SE(3) Group Convolution](https://github.com/ThijsKuipers1995/gconv) library. We deeply thank the authors for open sourcing their codes. We are also very grateful to the developers of the amazing libraries [torch geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html), [pytorch lightning](https://lightning.ai/), and [weights and biases](https://https://wandb.ai/) !
