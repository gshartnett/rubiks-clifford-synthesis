# Rubik's Cube Inspired Approach to Clifford Synthesis
Code repository for the paper "A Rubik's Cube Inspired Approach to Clifford Synthesis".

The problem of Clifford synthesis is: given an arbitrary element of the Clifford group Cl($n$), find a sequence of gates, drawn from some universal gate set, that implements the given Clifford element. Inspired by recent machine learning approaches for solving the Rubik's Cube, this repository contains research code used to develop a learned guidance function approach to Clifford synthesis.

## Contents
The key contents of this repository are:
- `clifford.py` contains code used for representing and manipulating Clifford group elements
- `lgf.py` contains code for defining the learned guidance function (lgf) neural network, as well as the greedy optimization of the lgf for Clifford synthesis
- `train.py` contains code for the training of the lgf on batches of randomly sampled Clifford elements
- `Data Analysis Notebook.ipynb` contains code used to analyze the numerical data generated in our experiments and to make tables and plots
- `Cl(2) Graph Notebook.ipynb` contains code used to exhaustively build the problem graph for the case of the $n=2$ Clifford group
- `job_manager.py` contains code used to set up many different training and evaluation runs

## Installation
`pip install -r requirements.txt`. It is recommended to perform the Python installation within a virtual environment.

## License
This code is provided under the MIT license. See `LICENSE` for more information.

## Citation
Add a bibtex reference
