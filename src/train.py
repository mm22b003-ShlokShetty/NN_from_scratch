import argparse
import numpy as np
import json
import wandb
import os

from ann.neural_network import NeuralNetwork
from ann.neural_layer import NeuralLayer
from ann.activations import ReLU, Sigmoid, Tanh
from ann.objective_functions import MeanSquaredError, CrossEntropyLoss
from ann.optimizers import SGD, Momentum, NAG, RMSProp
from utils.data_loader import load_data


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
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401_Assignment1")

    return parser.parse_args()


# -----------------------------
# Build Network
# -----------------------------
def build_network(args, input_dim=784, output_dim=10):

    model = NeuralNetwork()
    prev_dim = input_dim

    for i in range(args.num_layers):

        hidden_dim = args.hidden_size[i]
        model.add(NeuralLayer(prev_dim, hidden_dim, weight_init=args.weight_init))

        if args.activation == "relu":
            model.add(ReLU())
        elif args.activation == "sigmoid":
            model.add(Sigmoid())
        elif args.activation == "tanh":
            model.add(Tanh())
        else:
            raise ValueError("Invalid activation")

        prev_dim = hidden_dim

    model.add(NeuralLayer(prev_dim, output_dim, weight_init=args.weight_init))

    return model


# -----------------------------
# Choose Loss
# -----------------------------
def get_loss(name):

    if name == "mse":
        return MeanSquaredError()
    elif name == "cross_entropy":
        return CrossEntropyLoss()
    else:
        raise ValueError("Invalid loss function")


# -----------------------------
# Choose Optimizer
# -----------------------------
def get_optimizer(args):

    if args.optimizer == "sgd":
        return SGD(args.learning_rate, args.weight_decay)

    elif args.optimizer == "momentum":
        return Momentum(args.learning_rate, weight_decay=args.weight_decay)

    elif args.optimizer == "nag":
        return NAG(args.learning_rate, weight_decay=args.weight_decay)

    elif args.optimizer == "rmsprop":
        return RMSProp(args.learning_rate, weight_decay=args.weight_decay)

    else:
        raise ValueError("Invalid optimizer")


# -----------------------------
# Accuracy
# -----------------------------
def compute_accuracy(logits, y_true):

    preds = np.argmax(logits, axis=1)
    labels = np.argmax(y_true, axis=1)

    return np.mean(preds == labels)


# -----------------------------
# Training
# -----------------------------
def train(args):

    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        name=f"{args.optimizer}_lr{args.learning_rate}"
    )

    X_train, y_train, X_test, y_test = load_data(args.dataset)

    model = build_network(args)

    loss_fn = get_loss(args.loss)
    optimizer = get_optimizer(args)

    best_accuracy = 0.0
    best_weights = None

    for epoch in range(args.epochs):

        perm = np.random.permutation(X_train.shape[0])

        X_train = X_train[perm]
        y_train = y_train[perm]

        epoch_loss = 0

        for i in range(0, X_train.shape[0], args.batch_size):

            X_batch = X_train[i:i + args.batch_size]
            y_batch = y_train[i:i + args.batch_size]

            logits = model.forward(X_batch)

            loss = loss_fn.forward(logits, y_batch)
            epoch_loss += loss

            dOut = loss_fn.backward()

            model.backward(dOut)

            # ----- Gradient Norm (Q2.4) -----
            for layer in model.layers:
                if isinstance(layer, NeuralLayer):
                    grad_norm = np.linalg.norm(layer.grad_W)
                    break

            optimizer.step(model.layers)

        # ---------- Training Accuracy ----------
        train_logits = model.forward(X_train)
        train_acc = compute_accuracy(train_logits, y_train)

        # ---------- Test Accuracy ----------
        test_logits = model.forward(X_test)
        test_acc = compute_accuracy(test_logits, y_test)

        print(f"Epoch {epoch+1}/{args.epochs} - Test Accuracy: {test_acc:.4f}")

        # ----- Dead neuron analysis (Q2.5) -----
        activation_zero_fraction = None
        for layer in model.layers:
            if isinstance(layer, ReLU):
                activation_zero_fraction = np.mean(layer.Z <= 0)
                break

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "grad_norm_layer1": grad_norm,
            "dead_neuron_fraction": activation_zero_fraction
        })

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_weights = model.get_weights()

    # ----- Confusion Matrix (Q2.8) -----
    preds = np.argmax(test_logits, axis=1)
    labels = np.argmax(y_test, axis=1)

    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            preds=preds,
            y_true=labels
        )
    })

    os.makedirs("models", exist_ok=True)

    np.save("models/best_model.npy", np.array(best_weights), allow_pickle=True)

    with open("models/best_config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    print("Training complete. Best Accuracy:", best_accuracy)

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)