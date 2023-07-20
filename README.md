# A Rubik's Cube inspired approach to Clifford synthesis
This is the code repository for the paper "[A Rubik's Cube inspired approach to Clifford synthesis](https://arxiv.org/abs/2307.08684)".

The problem of Clifford synthesis is: given an arbitrary element of the Clifford group Cl($n$), find a sequence of gates drawn from some universal gate set that implements the given Clifford element. Inspired by recent machine learning approaches for solving the Rubik's Cube, this repository contains research code used to develop a learned guidance function approach to Clifford synthesis.

## Contents
The key contents of this repository are:
- `clifford.py` contains code used for representing and manipulating Clifford group elements
- `lgf.py` contains code for defining the learned guidance function (lgf) neural network, as well as the greedy optimization of the lgf for Clifford synthesis
- `train.py` contains code for the training of the lgf on batches of randomly sampled Clifford elements
- `Data Analysis Notebook.ipynb` contains code used to analyze the numerical data generated in our experiments and to make tables and plots
- `Cl(2) Graph Notebook.ipynb` contains code used to exhaustively build the problem graph for the case of the $n=2$ Clifford group

## Installation
This repo can be installed using [Poetry](https://python-poetry.org/):
```
git clone git@github.com:gshartnett/rubiks-clifford-synthesis.git
cd rubiks-clifford-synthesis
poetry install
```

It is recommended to perform the Python installation within a virtual environment. If your machine has a GPU, then you will have to manually install the appropriate [PyTorch](https://pytorch.org/get-started/locally/) version.

## License
This code is provided under the MIT license. See `LICENSE` for more information.

## Citation
If you'd like to cite our work, please use this bibtex citation:

```
@article{bao2023rubik,
  title={A Rubik's Cube inspired approach to Clifford synthesis},
  author={Bao, Ning and Hartnett, Gavin S},
  journal={arXiv preprint arXiv:2307.08684},
  year={2023}
}
```