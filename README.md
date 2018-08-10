# Neural network backpropagation from scratch in Python

The initial software is provided by the amazing tutorial "*How to Implement the Backpropagation Algorithm From Scratch In Python*" by Jason Brownlee.<br>
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

I git this soft because I add some features proposed by Jason Bronwlee in the "Extensions" part of his tutorial.

## About the training
*bacpropagation.py* implements a multilayer perceptron (MLP). This feedfoward network is trained and tested using k-fold cross-validation on *seeds_dataset.csv* dataset.<br>
As k = 5, 5 models are fit and evaluated on 5 different hold out sets. Each model is trained for 500 epochs.<br>
Dataset stands for wheat seeds. These inputs are normalized to the range (0, 1) by the code.<br>
The training process uses online gradient descent. The batch learning will soon be implemented.<br><br>
One hidden layer of 5 neurons and one output layer of 3 neurons are created to init a network. (No layer for inputs).<br>
*Note: Hidden layers could be added thanks to the custom init network function.*<br>
The number of neuron outputs is computed according to the number of class found in the dataset.<br>
They are translated into one-hot encoding to match the network outputs.<br>
Thanks to this, the error can be computed beetween expected outputs and predicted outputs.
