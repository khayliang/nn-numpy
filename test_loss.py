import numpy as np
import unittest
import loss

class TestHingeLoss(unittest.TestCase):
    def test_forward(self):
        loss_fn = loss.Hinge()
        preds = np.array([[-2, 1, -1, 2]])
        labels = np.array([[-1, 1, 1, -1]])
        expected_loss = np.array([[0, 0, 2, 3]])

        loss_vals = loss_fn.forward(preds, labels)
        self.assertTrue(np.array_equal(expected_loss, loss_vals))
    
    def test_backward(self):
        loss_fn = loss.Hinge()
        loss_fn.train()

        preds = np.array([[-2, 1, -1, 2]])
        labels = np.array([[-1, 1, 1, -1]])
        expected_grad = np.array([[0, 0, -1, 1]])

        loss_fn.forward(preds, labels)
        grad = loss_fn.backward(labels)
        self.assertTrue(np.array_equal(expected_grad, grad))


class TestNLL(unittest.TestCase):
    def test_forward(self):
        loss_fn = loss.NLL()
        preds = np.array([[0.4, 0.9, 0.1, 0.3]])
        labels = np.array([[1, 1, 0, 1]])
        expected_loss = np.array([[0.92, 0.11, 0.11, 1.2]])

        loss_vals = loss_fn.forward(preds, labels)
        loss_vals = np.around(loss_vals, 2)
        self.assertTrue(np.array_equal(expected_loss, loss_vals))
    
    def test_backward(self):
        loss_fn = loss.NLL()
        loss_fn.train()

        preds = np.array([[0.4, 0.9, 0.1, 0.3]])
        labels = np.array([[1, 1, 0, 1]])
        expected_grad = np.array([[-2.50, -1.11, 1.11, -3.33]])

        loss_fn.forward(preds, labels)
        grad = loss_fn.backward(labels)
        grad = np.around(grad, 2)

        self.assertTrue(np.array_equal(expected_grad, grad))

if __name__ == '__main__':
    unittest.main()
