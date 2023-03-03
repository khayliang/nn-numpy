import numpy as np
import activation
import loss
from sequential import Sequential
from linear import Linear

import data_gen

modules = [
    Linear(5, 7),
    activation.ReLU(),
    Linear(7, 7),
    activation.ReLU(),
    Linear(7, 5),
    activation.ReLU(),
    Linear(5, 3),
    activation.ReLU(),
    Linear(3, 1),
    ]
loss_fn = loss.Hinge()

model = Sequential(modules, loss_fn)

data = data_gen.create_random_data(5, 1000)
classifier = data_gen.create_polynomial_classifier(5, 5)
labels = classifier(data)
#labels[labels < 0] = 0

training_loss = model.sgd(data, labels, epochs=30)
print(np.around(training_loss, 2))
