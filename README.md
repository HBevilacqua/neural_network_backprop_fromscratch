# Neural network backpropagation from scratch in Python

The initial software is provided by the amazing tutorial "*How to Implement the Backpropagation Algorithm From Scratch In Python*" by Jason Brownlee.<br>
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

You should read this tuto which outlines the following steps:<br>
- Initialize Network
- Forward Propagation
- Backpropagation
- Train Network
- Predict
- Seeds Dataset Case Study

I git this soft to sum up what I've learned and add some features proposed by Jason Bronwlee in the "Extensions" part of his tutorial.<br>

To understand backpropagation calculations through a concrete example, take a look at *"A Step by Step Backpropagation Example*" by Matt Mazur:<br>
https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

<br>
*bacpropagation.py* implements a **multilayer feed foward neural network**.

## About the architecture of our neural network
#### Input layer
There is no activation function because we want to get the characteristics of the raw input vector.

#### Hidden layer
Five neurons are defined.<br>
*Note: Hidden layers could be added thanks to the custom init network function (cf. initialize_network_custom(tab)).*<br>
The **sigmoid** or **tanh** activation functions are available as a parameter of the evaluate_algorithm() function.

#### Output layer
They are three neurons. The number of output neurons is defined by the number of classes found in the dataset outputs. (Here, we are trying to solve a classification problem.) <br>
In classification problems, best results are achieved when the network has one neuron in the output layer for each class value.<br>
The output values are translated into **one-hot encoding** to match the network outputs.<br>
Our ouput layer uses the same activation function that the hidden layer (sigmoid or tanh).<br>
To **predict** the class which has the largest probability for one input vector, we are using the **arg max function**.<br><br>

## About the training (cost function and optimization algorithm)
The training process uses **Stochastic Gradient Descent** optimization algorithm. (SGD, called online machine learning algorithm as well).<br>
Note: The optimization is the mechanism which adjusts the weights to increase the accuracy of the predictions.<br>
This network is trained and tested using **k-fold cross-validation** on *seeds_dataset.csv* dataset.<br>
As k = 5, five models are fitted and evaluated on 5 different hold out sets. Each model is trained for **500 epochs**.<br>
The sum squared error between the expected output and the network output is accumulated each epoch.<br>
Dataset stands for wheat seeds. These inputs are normalized to the range (0, 1).<br><br>

### Glossary

#### Feedfoward neural network
Neural network without cycle between neurons (ex: no connection between layer N and layer N-2).

#### Feedfoward propagation
Computes output from a neural network by propagating input signals.

#### Gardient
The gradient (∇f) of a scalar-valued multivariable function f(x,y,…) gathers all its partial derivatives (
∂f/∂x, ∂f/∂Y, ...) into a vector.

#### Gardient descent 
It is a first order optimization algorithm to find the minimum of a function, generally used in ML when it is not possible to find the solutions of the equation ∂J(θ)/∂θ = 0 (J is the cost function), i.e. all θ which minimize J(θ).<br>
In this ML example, the gradient descent will find a local minimum according to the initial random weights allocated at the neural network initialization. The negative gradient computation shows us in which direction we have to update the weights.<br>
The GD is computed for each iteration by using: θ := θ - η.∇J(θ) (where η is the learning rate).

#### Loss function or Cost function
Error for a single training sample.
Square loss: J(ŷi,yi) = (ŷi - yi)^2 where ŷi = f(θ,b,xi) is the predicted output for the xi input.
Error for the entire training set.
Mean Squared Error: MSE(θ) = J(θ) = 1/N * ∑i=(1..N) (Ŷi - Yi)^2

#### Classification (in ML)
Classification aims to predict a label. The outputs are class labels.

#### Regression (in ML)
Regression aims to predict a quantity. The outputs are continuous.

#### Regression
Regression tries to predict outputs of a function according to its inputs (= find the relationship between Y and X).

#### Linear regression
Linear regression is a linear model, e.g. a model that assumes a linear relationship between the input variables (x) and the single output variable (y).

#### Backpropagation
Supervised method (gradient descent) to train networks, see the tuto above-mentioned for more details.

#### Training a network
Updates weights in a neural network to improve its predictions according to a dataset. Here, the SGD steps:<br>
<pre>
For each epoch
     For each train pattern
          Foward propagation (update the outputs: 'output')
          Back propagation (update the errors for each neuron: 'delta')
          Updating weights (update the weight according tot the errors: 'weights')
</pre>
#### Dataset
Data used to train and test the network.

#### Arg max
The argument of the maxima refers to the inputs where a function output values are the highest.<br>
(e.g. fox x ∈ [0,π], the *arg max* of sin(x) is x = π/2 and the *max* of sin(x) is sin(x) = 1.

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

#### Epoch
One epoch = One cycle (foward + backward) through the entire training dataset (all the rows "inputs/outputs" seen).

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
