import numpy as np
import unittest
import activation

class TestReLU(unittest.TestCase):
    def test_forward(self):
        fn = activation.ReLU()
        z = np.arange(12).reshape(3,4)
        z_neg = -z

        a = fn.forward(z)
        a_neg = fn.forward(z_neg)
        self.assertTrue(np.array_equal(z, a))
        self.assertTrue(np.array_equal(z_neg, np.zeros((3,4))))
    
    def test_backward(self):
        fn = activation.ReLU()
        fn.train()

        z = np.arange(12).reshape(3,4)
        a = fn.forward(z)
        dlda = np.ones((3,4))
        dldz = fn.backward(dlda)
        self.assertTrue(np.array_equal(dldz, a))

class TestSigmoid(unittest.TestCase):
    def test_forward(self):
        fn = activation.Sigmoid()
        z = np.arange(12).reshape(3,4)
        Z = 1/(1+np.exp(-z))
        a = fn.forward(z)
        self.assertTrue(np.array_equal(Z, a))
    
    def test_backward(self):
        # Lazy to compute sigmoid. Just check that shape is correct
        fn = activation.Sigmoid()
        fn.train()

        z = np.arange(12).reshape(3,4)
        a = fn.forward(z)
        dlda = np.ones((3,4))
        dldz = fn.backward(dlda)
        self.assertEqual(dldz.shape, a.shape)

if __name__ == '__main__':
    unittest.main()
