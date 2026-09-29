"""Minimal single-objective Deep Feature Selection (DFS) example."""
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DFS(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # One learnable multiplier for each input feature.
        self.input_weights = nn.Parameter(torch.ones(input_dim))
        self.hidden1 = nn.Linear(input_dim, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 7)

        # Match the baseline's initialization.
        for layer in (self.hidden1, self.hidden2):
            bound = math.sqrt(6 / (layer.in_features + layer.out_features))
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = x * self.input_weights
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        return self.output(x)  # CrossEntropyLoss takes logits directly.


def main():
    torch.manual_seed(1000)
    root = Path(__file__).resolve().parent
    X = np.loadtxt(root / 'GM12878_200bp_Data.txt', dtype=np.float32)
    labels = np.loadtxt(root / 'GM12878_200bp_Classes.txt', dtype=str)
    y = LabelEncoder().fit_transform(labels)

    # Normalize each sample independently; keep all-zero rows unchanged.
    norms = np.linalg.norm(X.astype(np.float64), axis=1, keepdims=True)
    X = (X / np.where(norms > 0, norms, 1)).astype(np.float32)
    input_dim = X.shape[1]
    print(f"Detected input dimension: {input_dim}")

    # Approximately equal thirds, preserving class proportions in each split.
    x_train, x_rest, y_train, y_rest = train_test_split(
        X, y, test_size=2 / 3, stratify=y, random_state=1000)
    x_val, x_test, y_val, y_test = train_test_split(
        x_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=1000)

    train_data = TensorDataset(torch.from_numpy(x_train),
                               torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(train_data, batch_size=100, shuffle=True)
    x_val, x_test = torch.from_numpy(x_val), torch.from_numpy(x_test)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    model = DFS(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.1)
    lambda1, alpha1 = 0.01, 0.0001

    for epoch in range(1, 1001):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            # L1 encourages small feature weights; L2 regularizes dense weights.
            # Biases are excluded from both penalties.
            l1 = model.input_weights.abs().sum()
            l2 = sum(layer.weight.square().sum()
                     for layer in (model.hidden1, model.hidden2, model.output))
            loss = criterion(model(xb), yb) + lambda1 * l1 + (alpha1 / 2) * l2
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_accuracy = (model(x_val).argmax(1) == y_val).float().mean().item()
        print(f'Epoch {epoch:4d}: validation accuracy = {val_accuracy:.4f}')

    # Evaluate the final trained model on the held-out test set.
    model.eval()
    with torch.no_grad():
        test_accuracy = (model(x_test).argmax(1) == y_test).float().mean().item()
    print(f'Final test accuracy = {test_accuracy:.4f}')


if __name__ == '__main__':
    main()
