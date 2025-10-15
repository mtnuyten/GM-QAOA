# Variance of the GM-QAOA loss function for the MaxCut problem

This code was used for the numerical simulations found in [arXiv](https://arxiv.org/abs/2509.10424).

The authors derive an explicit formula for the loss function variance and prove that for broad optimization problems (including MaxCut, SAT, TSP), GM-QAOA avoids barren plateaus for sufficiently deep circuits by establishing an inverse-polynomial lower bound on the variance. The goal of this project was to simulate the QAOA circuit with Grover mixer for the MaxCut problem and provide evidence of the variance of the loss function stays above this analytical lower bound. 

## Launch with Binder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mtnuyten/GM-QAOA/HEAD?urlpath=%2Fdoc%2Ftree%2FNumerical+Simulations+for+MaxCut.ipynb)

or follow these steps to get the code running.
### Prerequisites
You will need to have Anaconda or Miniconda installed to manage the environment.

### Installation

1.  Clone the repo:
    ```sh
    git clone [https://github.com/mtnuyten/GM-QAOA.git](https://github.com/mtnuyten/GM-QAOA.git)
    cd GM-QAOA
    ```
2.  Create and activate the conda environment using the provided file:
    ```sh
    conda env create -f necessary_packages.yml
    conda activate quantum-env
    ```
    *(Note: The environment name, `quantum-env`, is specified inside the `necessary_packages.yml` file.)*

## Necessary Packages

This project was built using a number of fantastic open-source libraries, including:

* [PennyLane](https://pennylane.ai/) - A cross-platform Python library for quantum machine learning.
* [NumPy](https://numpy.org/) - The fundamental package for scientific computing with Python.
* [Matplotlib](https://matplotlib.org/) - A comprehensive library for creating static, animated, and interactive visualizations in Python.

