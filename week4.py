"""
@author: Jinal Shah

This file will implement
a simple 3 node neural network
from scratch. This is the code
from Week 4 of my Deep Learning
with PyTorch series on Medium.

"""
# Importing all libraries
import numpy as np


# Building the NN
class NeuralNetwork:
    # Constructor
    def __init__(self):
        # Defining the layers
        self.weights1 = np.random.random(size=(2, 1))
        self.weights2 = np.random.random(size=(2, 1))

        # Defining the biases
        self.b1 = np.array(1).reshape(1, 1)
        self.b2 = np.array(1).reshape(1, 1)

    # Forward function
    # Note, I am assuming that X doesn't need any transformations such as flattening
    def forward(self, X):
        y = X.copy()

        # Putting input through hidden layer 1
        y = np.concatenate((self.b1, y), axis=1)
        y = np.matmul(y, self.weights1)
        y = self.sigmoid(y)

        # Taking output from hidden1 and delivering to output
        y = np.concatenate((self.b2, y), axis=1)
        y = np.matmul(y, self.weights2)

        # Returning result
        return y

    # Sigmoid Function
    def sigmoid(self, X):
        return 1 / (1 + np.power(np.e, -X))

