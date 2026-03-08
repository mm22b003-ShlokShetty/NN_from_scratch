import numpy as np


class SGD:
    def __init__(self, learning_rate, weight_decay=0.0):
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def step(self, layers):
        for layer in layers:
            if hasattr(layer, "W"):
                grad_W = layer.grad_W + self.weight_decay * layer.W
                grad_b = layer.grad_b

                layer.W -= self.lr * grad_W
                layer.b -= self.lr * grad_b


class Momentum:
    def __init__(self, learning_rate, beta=0.9, weight_decay=0.0):
        self.lr = learning_rate
        self.beta = beta
        self.weight_decay = weight_decay
        self.v = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            if hasattr(layer, "W"):

                if i not in self.v:
                    self.v[i] = {
                        "W": np.zeros_like(layer.W),
                        "b": np.zeros_like(layer.b)
                    }

                grad_W = layer.grad_W + self.weight_decay * layer.W
                grad_b = layer.grad_b

                self.v[i]["W"] = self.beta * self.v[i]["W"] + (1 - self.beta) * grad_W
                self.v[i]["b"] = self.beta * self.v[i]["b"] + (1 - self.beta) * grad_b

                layer.W -= self.lr * self.v[i]["W"]
                layer.b -= self.lr * self.v[i]["b"]


class NAG:
    def __init__(self, learning_rate, beta=0.9, weight_decay=0.0):
        self.lr = learning_rate
        self.beta = beta
        self.weight_decay = weight_decay
        self.v = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            if hasattr(layer, "W"):

                if i not in self.v:
                    self.v[i] = {
                        "W": np.zeros_like(layer.W),
                        "b": np.zeros_like(layer.b)
                    }

                grad_W = layer.grad_W + self.weight_decay * layer.W
                grad_b = layer.grad_b

                prev_v_W = self.v[i]["W"]
                prev_v_b = self.v[i]["b"]

                self.v[i]["W"] = self.beta * self.v[i]["W"] + (1 - self.beta) * grad_W
                self.v[i]["b"] = self.beta * self.v[i]["b"] + (1 - self.beta) * grad_b

                layer.W -= self.lr * (
                    self.beta * prev_v_W + (1 - self.beta) * grad_W
                )
                layer.b -= self.lr * (
                    self.beta * prev_v_b + (1 - self.beta) * grad_b
                )


class RMSProp:
    def __init__(self, learning_rate, beta=0.9, eps=1e-8, weight_decay=0.0):
        self.lr = learning_rate
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay
        self.s = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            if hasattr(layer, "W"):

                if i not in self.s:
                    self.s[i] = {
                        "W": np.zeros_like(layer.W),
                        "b": np.zeros_like(layer.b)
                    }

                grad_W = layer.grad_W + self.weight_decay * layer.W
                grad_b = layer.grad_b

                self.s[i]["W"] = self.beta * self.s[i]["W"] + (1 - self.beta) * (grad_W ** 2)
                self.s[i]["b"] = self.beta * self.s[i]["b"] + (1 - self.beta) * (grad_b ** 2)

                layer.W -= self.lr * grad_W / (np.sqrt(self.s[i]["W"]) + self.eps)
                layer.b -= self.lr * grad_b / (np.sqrt(self.s[i]["b"]) + self.eps)