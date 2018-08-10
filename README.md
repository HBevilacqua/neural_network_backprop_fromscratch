# Neural network backpropagation from scratch in Python

The initial software is provided by the amazing tutorial "*How to Implement the Backpropagation Algorithm From Scratch In Python*" by Jason Brownlee.<br>
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

If you find my repository, I advise you to follow this tuto which outlines each step before reading my code:<br>
- Initialize Network
- Forward Propagate
- Back Propagate Error
- Train Network
- Predict
- Seeds Dataset Case Study

I git this soft because I add some features proposed by Jason Bronwlee in the "Extensions" part of his tutorial.

## About the neural network training used
*bacpropagation.py* implements a multilayer perceptron (MLP). This feedfoward network is trained and tested using k-fold cross-validation on *seeds_dataset.csv* dataset.<br>
As k = 5, 5 models are fit and evaluated on 5 different hold out sets. Each model is trained for 500 epochs.<br>
Dataset stands for wheat seeds. These inputs are normalized to the range (0, 1) by the code.<br>
The training process uses online gradient descent. The batch learning will soon be implemented.<br><br>
One hidden layer of 5 neurons and one output layer of 3 neurons are created to init a network. (No layer for inputs).<br>
*Note: Hidden layers could be added thanks to the custom init network function.*<br>
The number of neuron outputs is computed according to the number of class found in the dataset.<br>
They are translated into one-hot encoding to match the network outputs.<br>
Thanks to this, the error can be computed beetween expected outputs and predicted outputs.

## Glossary

#### MLP - Multilayer perceptron
a type of feedforward artificial neural network.

#### Feedfoward neural network
Neural network whitout cycle between neurons (ex: no connection between layer N and layer N-2).

#### Feedfoward propagation
Compute output from a neural network by propagating input signals.

#### Backpropagation
Supervised method (gradient descent) to train networks, see the tuto above-mentioned for more details.

#### Training a network
Update weights in a neural network to improve its predictions according to a dataset. Here, the steps in our case:<br>
Loop while not trained enough:
1. foward propagation
2. back propagation
3. updating weights

#### Dataset
Data used to train and test the network.

#### k-cross validation
It is a procedure used to estimate the skill of the model on new data.<br>
k refers to the number of groups that a given data sample is to be split into.
Sequence:
1. Shuffle the dataset randomly.
2. Split the dataset into k groups
3. For each unique group:<br>
     A. Take the group as a hold out or test data set<br>
     B. Take the remaining groups as a training data set<br>
     C. Fit a model on the training set and evaluate it on the test set<br>
     D. Retain the evaluation score and discard the model<br>
4. Summarize the skill of the model using the sample of model evaluation scores

#### One-hot encoding
A new unic binary variable is added for each integer value:<br>
<pre>
red,  green,  blue<br>
1,    0,    0<br>
0,    1,    0<br>
0,    0,    1<br>
</pre>

#### Integer encoding
“red” is 1, “green” is 2, and “blue” is 3.
