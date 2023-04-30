# Rubik's Cube Inspired Approach to Clifford Synthesis
Code repository for the paper "A Rubik's Cube Inspired Approach to Clifford Synthesis".

## Introduction
Blah

## Contents
The key contents of this repository are:

## Installation
`pip install -r requirements.txt`. It is recommended to perform the Python installation within a virtual environment.

## License
This code is provided under the MIT license. See `LICENSE` for more information.

## TODO
- Code
    - clean up notebooks
    - fill out README
    - try to vectorize clifford composition when phase bits are included
    - add GPU support
- LGF training
    - play around with NN architecture
    - play around with data generation/random walk
    - possible: investigate other loss functions
- for the case where the phase bits are dropped, some of the functionality is incorrectly using the phase bits
- possible: implement the genetic algorithm approach
- possible: code up Djikstra if possible
- generate results and add to paper
- consider adding tensorboard logger

## Citation
Add a bibtex reference