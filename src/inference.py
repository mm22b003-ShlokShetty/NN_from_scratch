import argparse
import numpy as np
import json

from ann.neural_network import NeuralNetwork
from ann.neural_layer import NeuralLayer
from ann.activations import ReLU, Sigmoid, Tanh
from utils.data_loader import load_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -----------------------------
# SAME CLI AS train.py
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", type=str, default="mnist")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy")
    parser.add_argument("-o", "--optimizer", type=str, default="sgd")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", nargs="+", type=int, default=[128])
    parser.add_argument("-a", "--activation", type=str, default="relu")
    parser.add_argument("-wi", "--weight_init", type=str, default="random")
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401")

    parser.add_argument("--model_path", type=str, default="models/best_model.npy")
    parser.add_argument("--config_path", type=str, default="models/best_config.json")

    return parser.parse_args()


# -----------------------------
# Build Network
# -----------------------------
def build_network(config, input_dim=784, output_dim=10):
    model = NeuralNetwork()
    prev_dim = input_dim

    for i in range(config["num_layers"]):
        hidden_dim = config["hidden_size"][i]
        model.add(NeuralLayer(prev_dim, hidden_dim, weight_init=config["weight_init"]))

        if config["activation"] == "relu":
            model.add(ReLU())
        elif config["activation"] == "sigmoid":
            model.add(Sigmoid())
        elif config["activation"] == "tanh":
            model.add(Tanh())
        else:
            raise ValueError(f"Invalid activation: {config['activation']}")

        prev_dim = hidden_dim

    # Output layer (logits, no activation)
    model.add(NeuralLayer(prev_dim, output_dim, weight_init=config["weight_init"]))

    return model


# -----------------------------
# Load saved weights
# -----------------------------
def load_weights(path):
    data = np.load(path, allow_pickle=True)
    if data.ndim == 0:
        return data.item()
    return data


# -----------------------------
# Inference
# -----------------------------
def run_inference(args):

    # Load config saved during training (source of truth for architecture)
    with open(args.config_path) as f:
        config = json.load(f)

    # Load test data using dataset from saved config
    _, _, X_test, y_test = load_data(config["dataset"])

    # Rebuild model architecture from saved config
    model = build_network(config)

    # Load and restore saved weights
    weights = load_weights(args.model_path)
    model.set_weights(weights)

    # Forward pass
    logits = model.forward(X_test)

    preds = np.argmax(logits, axis=1)
    labels = np.argmax(y_test, axis=1)

    acc       = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro")
    recall    = recall_score(labels, preds, average="macro")
    f1        = f1_score(labels, preds, average="macro")

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)