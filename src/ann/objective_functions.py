import numpy as np


class MeanSquaredError:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        """
        Forward pass for MSE loss
        """
        self.y_pred = y_pred
        self.y_true = y_true

        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    def backward(self):
        """
        Backward pass for MSE loss
        Returns gradient w.r.t. predictions
        """
        N = self.y_true.shape[0]
        dA = (2 / N) * (self.y_pred - self.y_true)
        return dA


class CrossEntropyLoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        """
        Forward pass for Cross Entropy loss
        Assumes y_pred are probabilities (after softmax)
        """
        self.y_pred = y_pred
        self.y_true = y_true

        # Numerical stability
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

        loss = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
        return loss

    def backward(self):
        """
        Backward pass for Cross Entropy loss
        Returns gradient w.r.t. logits (Softmax + CE combined)
        """
        N = self.y_true.shape[0]
        dZ = (self.y_pred - self.y_true) / N
        return dZ