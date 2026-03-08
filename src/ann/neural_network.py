class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dOut):
        for layer in reversed(self.layers):
            dOut = layer.backward(dOut)
        return dOut
    
    def get_weights(self):
        weights = []
        for layer in self.layers:
            if hasattr(layer, "W"):
                weights.append({
                    "W": layer.W,
                    "b": layer.b
                })
        return weights

    def set_weights(self, weights):
        dense_idx = 0
        for layer in self.layers:
            if hasattr(layer, "W"):
                layer.W = weights[dense_idx]["W"]
                layer.b = weights[dense_idx]["b"]
                dense_idx += 1