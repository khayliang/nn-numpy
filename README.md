# nn-numpy
The goal of this notebook is to implement a basic neural network using only `numpy`, and explore how the depth (no. of layers) and breadth (no. of units in a layer) of a neural network enables it to better classify to non-linear datasets.

The notebook is divided into two parts:
- Implementation: code for the neural network
- Results: analysis of neural network configuration for fitting non-linear datasets

## Todo
- decouple the thresholding of output for classification from the spaghetti
- add negative log likelihood
- use sigmoid for output layer

# Neural Network Implementation
For the implementation of the neural network, we will implement the following features only since we are only interested in classification capabilities:

- Activation functions: ReLU, Linear, Sigmoid
- Loss function: Hinge
- Gradient descent method: Stochastic gradient descent

The neural network will also be lacking the following features:
- Regularization
- Regression

The limitations of the following implementation are:
- only able to use labels `[-1, 1]` because only hinge loss is implemented. 
- only able to use linear activation function for output layer 
