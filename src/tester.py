from utils.data_loader import load_data

X_train, y_train, X_test, y_test = load_data("mnist")

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)