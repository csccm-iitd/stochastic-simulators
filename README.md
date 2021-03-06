# Overview
[A deep learning based surrogate model for stochastic simulators](https://arxiv.org/abs/2110.13809)\
Akshay Thakur and [Souvik Chakraborty](https://www.csccm.in/home)\
\
TensorFlow implementation of deep learning-based surrogate model for stochastic simulators. Generative neural network is used to approximate the stochastic response. A simple feed-forward neural network is used with a conditional maximum mean discrepancy (CMMD) loss-function. CMMD allows to capture the discrepancy between the true response of the stochastic simulator and the distribution predicted by the neural network.

<p>
    <img src="Images/Neural Net.png" width="1040" height="440" />
</p>
Figure:  Schematic representation of the proposed deep learning framework based on CGMMN.

# Dependencies

- tensorflow 2.8.0
- python 3.x
- numpy
- matplotlib
- pandas
- scipy
- scikit-learn

# Installation
- Install TensorFlow and other dependencies.
- Clone the repository using
```
git clone https://github.com/name_add/Deep-Learning-Based-Surrogate-Model-for-Stochastic-Simulator.git
```
# Dataset for Training
The dataset for the problems of SDE without closed form solution and Stochastic SIR could be generated using the code in **Data** folder of this repository.
For the remaining two problems of 1-D and 2-D Stochastic Simulator problems with closed form solution, the code for dataset generation is inside the respective python scripts.

# Citation
If this respository helped you, please considering the citing the following work:
