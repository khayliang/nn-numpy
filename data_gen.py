import numpy as np

def create_random_linear_classifier(dim):
    w = np.random.uniform(-3, 3, size=(dim, 1))
    def classifier(x):
        y = w.T.dot(x)
        y[y > 0] = 1
        y[y <= 0] = -1
        return y
    return classifier

def create_polynomial_classifier(dim, deg):
    linear_classifier = create_random_linear_classifier(dim*deg)
    def classifier(x):
        poly_data = create_polynomial_data(x, deg)
        return linear_classifier(poly_data)
    return classifier

def create_polynomial_data(x, deg):
    """
    Increase the dimensionality of dataset
    [x_1, ..., x_n, x_1^2, ..., x_n^2, ..., x_1^d, ..., x_n^d]
    
    :params x: data to be processed in shape (d, n)
    :params deg: polynomial degree of classifier
    """

    d, n = x.shape
    new_x = [x]
    for _ in range(deg - 1):
        prev_x = x
        new_x.append(np.multiply(x, prev_x))
    new_x = np.array(new_x)
    new_x = new_x.reshape((deg*d, n))
    return new_x

def create_random_data(dim, n):
    return np.random.uniform(-5, 5, size=(dim, n))