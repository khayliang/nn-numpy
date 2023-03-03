import numpy as np

from module import Module

class Linear(Module):
    def __init__(self, m, n, weights=None, gain=None):
        """
        Initialize a Layer object.
        
        :params m: input vec length
        :params n: number of units in layer
        :params weights: mxn initial weights
        :params gain: nx1 initial gain
        """
        super().__init__()

        self.m, self.n = (m, n)

        if isinstance(weights, np.ndarray): 
            self.weights = weights
        else:
            self.weights = np.random.normal(0, 1.0 * m ** (-0.5), (m, n))

        if isinstance(gain, np.ndarray):
            self.gain = gain
        else:
            self.gain = np.zeros((self.n, 1))

        self.input = None
        #shape kxmxn
        self.dldw = None
        #shape kxn
        self.dldw0 = None

    def forward(self, a):
        """
        Forward pass for unit layer.

        :params a: mxk input vector
        :returns: nxk vector
        """
        if self.training:
            self.input = a
        output = self.weights.T.dot(a) + self.gain
        return output
    
    def gd_step(self, lrate):
        """
        Take gradient descent step via learning rate

        :params lrate: scalar of learning rate
        """
        k, m, n = self.dldw.shape
        dldw_ave = np.sum(self.dldw, axis=0)/k
        dldw0_ave = np.sum(self.dldw0, axis=0)/k

        self.weights = self.weights - lrate*dldw_ave
        self.gain = self.gain - lrate*dldw0_ave


    
    def backward(self, dldz):
        """
        Backward pass for unit layer. calculate derivatives and return dlda

        :params dldz: nxk vector
        :returns: mxk vector
        """
        if not self.training:
            raise Exception("Not in training mode. Backwards pass not allowed")

        n, k = dldz.shape
        dldz = dldz.reshape((k, n, 1))

        m, k = self.input.shape
        a = self.input.reshape((k, 1, m))

        self.dldw = np.einsum('kij,kjl->kli', dldz, a)
        self.dldw0 = dldz

        dlda = np.einsum("ij,kjl->kil", self.weights, dldz)
        dlda = dlda.reshape(m, k)
        return dlda
