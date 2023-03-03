import numpy as np

from module import Module

class Loss(Module):
    def __init__(self):
        super().__init__()
        self.output = None

class Hinge(Loss):
    def forward(self, a, labels):
        """
        Hinge loss forward pass
        
        :params a: 1xk vector of predictions for all input data
        :params labels: 1xk vector of labels of {-1, 1}
        :returns: 1xk vector of loss by each input data
        """
        x = 1 - a*labels
        x[x < 0] = 0
        if self.training:
            self.output = x
        return x
    
    def backward(self, labels):
        """
        Hinge loss backward pass. Computes gradient dLda
        
        :params labels: 1xk vector of labels of {-1, 1}
        :returns: 1xk vector of dLda
        """
        if not self.training:
            raise Exception("Not in training mode. Backwards pass not allowed")

        l = self.output
        l[l > 0] = 1
        dlda = -l*labels
        return dlda

class NLL(Loss):
    def forward(self, a, labels):
        """
        Negative log likelihood loss forward pass
        
        :params a: 1xk vector of predictions for all input data
        :params labels: 1xk vector of labels of {0, 1}
        :returns: 1xk vector of loss by each input data
        """
        loss = -(labels*np.log(a) + (1-labels)*np.log(1-a))
        if self.training:
            self.input = a
        return loss
    
    def backward(self, labels):
        """
        Negative log likelihood loss backward pass. Computes gradient dLda
        
        :params labels: 1xk vector of labels of {0, 1}
        :returns: 1xk vector of dLda
        """
        if not self.training:
            raise Exception("Not in training mode. Backwards pass not allowed")

        a = self.input
        dlda = (1-labels)/(1-a) - labels/a
        return dlda