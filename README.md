# Neural network backpropagation from scratch in Python

The initial code is provided by the amazing tutorial "*How to Implement the Backpropagation Algorithm From Scratch In Python*" by Jason Brownlee.<br>
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

I git this code to understand mechanisms before starting projects using machine learning libraries. I add some comments according to my understanding.<br>
I add some features proposed by Jason Bronwlee in the "Extensions" part of its tutorial.

## About this tuto
*bacpropagation.py* implements a multilayer perceptron (MLP). This feedfoward network is trained and tested using k-fold cross-validation on *seeds_dataset.csv* dataset.<br>
Dataset stands for wheat seeds. These inputs are normalized to the range (0, 1).<br>
The training process uses online gradient descent. The batch learning will be implemented.




