import numpy as np
import unittest
from linear import Linear

class TestLinear(unittest.TestCase):
    def test_forward(self):
        weights = np.ones((2, 5))
        gain = np.ones((5, 1))
        linear = Linear(2, 5, weights, gain)

        a = np.array([[1,0]]).T
        z = linear.forward(a)

        expected_z = np.ones((5,1))*2
        self.assertTrue(np.array_equal(z, expected_z))

    def test_backward(self):
        weights = np.ones((2, 5))
        gain = np.ones((5, 1))
        linear = Linear(2, 5, weights, gain)
        linear.train()

        a = np.array([[1,0]]).T
        z = linear.forward(a)

        dldz = np.array([[1, 0, 1, 0, 1]]).T
        dlda = linear.backward(dldz)

        expected_dlda = np.array([[3,3]]).T
        self.assertTrue(np.array_equal(dlda, expected_dlda))

        expected_dldw = np.array([[[1,0,1,0,1], [0,0,0,0,0]]])
        self.assertTrue(np.array_equal(linear.dldw, expected_dldw))
        expected_dldw0 = np.array([dldz])
        self.assertTrue(np.array_equal(linear.dldw0, expected_dldw0))

    def test_gd_step(self):
        weights = np.ones((2, 5))
        gain = np.ones((5, 1))
        linear = Linear(2, 5, weights, gain)

        dldz = np.array([[1, 0, 1, 0, 1]]).T
        expected_dlda = np.array([[3,3]]).T
        expected_dldw = np.array([[[1,0,1,0,1], [0,0,0,0,0]]])
        expected_dldw0 = np.array([dldz])

        linear.dldw = expected_dldw
        linear.dldw0 = expected_dldw0

        linear.gd_step(1)
        expected_weights = weights - np.array([[1,0,1,0,1], [0,0,0,0,0]])
        expected_gain = gain - np.array(dldz)

        self.assertTrue(np.array_equal(linear.weights, expected_weights))
        self.assertTrue(np.array_equal(linear.gain, expected_gain))



if __name__ == '__main__':
    unittest.main()
