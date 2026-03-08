import numpy as np


class NeuralLayer:
    def __init__(self, input_dim, output_dim, weight_init="random"):
        if weight_init == "random":
            self.W = np.random.randn(input_dim, output_dim) * 0.01

        elif weight_init == "xavier":
            limit = np.sqrt(2.0 / (input_dim + output_dim))
            self.W = np.random.randn(input_dim, output_dim) * limit

        elif weight_init == "zeros":
            self.W = np.zeros((input_dim, output_dim))

        else:
            raise ValueError("weight_init must be 'random', 'xavier', or 'zeros'")

        self.b = np.zeros(output_dim)

        self.grad_W = None
        self.grad_b = None

        self.X = None

    def forward(self, X):
        self.X = X
        Z = X @ self.W + self.b
        return Z

    def backward(self, dZ):
        self.grad_W = self.X.T @ dZ
        self.grad_b = np.sum(dZ, axis=0)
        dX = dZ @ self.W.T
        return dX