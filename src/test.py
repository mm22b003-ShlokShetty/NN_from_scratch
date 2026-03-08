from ann.neural_layer import NeuralLayer
from ann.optimizers import RMSProp
import numpy as np

layer = NeuralLayer(4, 2)
opt = RMSProp(learning_rate=0.01)

X = np.random.randn(5, 4)
Z = layer.forward(X)
dZ = np.random.randn(5, 2)
layer.backward(dZ)

W_before = layer.W.copy()
opt.step([layer])
W_after = layer.W.copy()

print("Weights changed:", not np.allclose(W_before, W_after))