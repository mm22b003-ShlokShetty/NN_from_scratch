import numpy as np


class ReLU:
    def __init__(self):
        self.Z = None

    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        dZ = dA * (self.Z > 0)
        return dZ


class Sigmoid:
    def __init__(self):
        self.A = None

    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dA):
        dZ = dA * self.A * (1 - self.A)
        return dZ


class Tanh:
    def __init__(self):
        self.A = None

    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dA):
        dZ = dA * (1 - self.A ** 2)
        return dZ