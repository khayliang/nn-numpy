import numpy as np
from module import Module

class Sequential:
    def __init__(self, modules, loss):
        self.children = modules
        self.loss = loss
    
    def sgd(self, X, Y, epochs=10, lrate=0.005):
        m, k = X.shape
        training_loss = []

        for child in self.children:
            child.train()
        self.loss.train()
        for i in range(epochs):
            loss = 0
            for i, x in enumerate(X.T):
                x = np.array([x]).T
                y = np.array([[Y[0,i]]])
                pred = self.forward(x)
                loss += self.loss.forward(pred, y)

                self.backward(y)
                self.gd_step(lrate)
            training_loss.append(loss/k)
        return training_loss

    def forward(self, data):
        """
        forward pass to calculate prediction

        :params data: mxk vector of k input data
        :returns: nxk vector of predictions
        """
        x = data
        for child in self.children:
            x = child.forward(x)
        return x
    def backward(self, labels):
        """
        backward pass to calculate gradients

        :params labels: 1xk vector of corresponding labels to data 
        """
        grad = self.loss.backward(labels)
        for child in reversed(self.children):
            grad = child.backward(grad)
        return True
    
    def gd_step(self, lrate):
        """
        update weights according to backward pass calculations

        :params lrate: scalar for learning rate
        """
        for child in self.children:
            child.gd_step(lrate)
        return True