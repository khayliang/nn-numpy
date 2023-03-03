import numpy as np

from module import Module

class Activation(Module):
    def __init__(self):
        super().__init__()
        self.output = None
        self.input = None

    def dadz(self):
        """
        Compute gradient dadz
        
        :returns: dadz, mxmxk vector
        """
        raise Exception("dadz unimplemented on Activation parent class")
    
    def backward(self, dlda):
        """
        Backwards pass. Returns dLdZ
        
        :params dlda: vector of shape mxk 
        :returns: dldz, mxk vector
        """
        if not self.training:
            raise Exception("Not in training mode. Backwards pass not allowed")

        dadz = self.dadz(self.output)

        m, k = dlda.shape

        dlda = dlda.reshape(k, m, 1)
        dldz = np.einsum('kij,kil->kil',dadz, dlda)
        dldz = dldz.reshape(m,k)
        return dldz

class ReLU(Activation):
    def forward(self, Z):
        """
        Forward pass. Runs matrix Z with ReLU function

        :params Z: mxk vector
        :returns: mxk vector
        """
        Z[Z < 0] = 0
        if self.training:
            self.output = Z
        return Z
    
    def dadz(self, a):
        a[a > 0] = 1
        m, k = a.shape
        a = a.reshape((k, 1, m))
        dadz = np.eye(m)*a
        return dadz

class Sigmoid(Activation):
    def forward(self, Z):
        """
        Forward pass. Runs matrix Z with sigmoid function

        :params Z: mxk vector
        :returns: mxk vector
        """
        Z = 1/(1+np.exp(-Z))
        if self.training:
            self.output = Z
        return Z

    def dadz(self, a):
        a = a*(1-a)

        m, k = a.shape
        a = a.reshape((k, 1, m))
        dadz = np.eye(m)*a
        return dadz

class Tanh(Activation):
    pass

class SoftMax(Activation):
    pass