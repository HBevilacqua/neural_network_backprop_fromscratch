# Neural network backpropagation from scratch in Python

The initial software is provided by the amazing tutorial "*How to Implement the Backpropagation Algorithm From Scratch In Python*" by Jason Brownlee.<br>
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

I git this soft because I add some features proposed by Jason Bronwlee in the "Extensions" part of his tutorial.

## About this tuto
*bacpropagation.py* implements a multilayer perceptron (MLP). This feedfoward network is trained and tested using k-fold cross-validation on *seeds_dataset.csv* dataset.<br>
As k = 5, 5 models are fit and evaluated on 5 different hold out sets. Each model is trained for 500 epochs.<br>
Dataset stands for wheat seeds. These inputs are normalized to the range (0, 1) by the code.<br>
The training process uses online gradient descent. The batch learning will soon be implemented.<br><br>

One hidden and one output layers are created. (No layer for inputs).<br>
Hidden layers could be added thanks to the custom init network function.<br>
Each layer contains neutron arrays of weights. The length of weights is the number of neuron input + 1 (the bias).<br>
The number of outputs is computed according to the number of class found in the dataset.<br>
They are translated into one-hot encoding to match the network outputs, thus the error can be calculated.
