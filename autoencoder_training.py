import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import plotly.graph_objects as go
import json
import os

def loader(weights_filepath, params_filepath):
    weights = torch.load(weights_filepath, weights_only=True, map_location=torch.device('cpu'))
    with open(params_filepath, "r") as f:
        params = json.load(f)
    return weights, params

def generate_data(alphabet_size, total_size):
    input = np.random.randint(0, alphabet_size, total_size)
    input_one_hot = np.eye(alphabet_size)[input]
    input_windows = input.reshape(-1, window_size)
    target_data = input_windows[:, window_size // 2]

    # Convert data to PyTorch tensors.
    input_tensor = torch.tensor(input_one_hot, dtype=torch.float32)
    target_tensor = torch.tensor(target_data, dtype=torch.long)

    return input_tensor, target_tensor


def compute_spread_loss(tx_symbols, sample_size=1000, epsilon=1e-3):
    if tx_symbols.size(0) < 2:
        return torch.tensor(0.0, device=tx_symbols.device)

    if tx_symbols.size(0) > sample_size:
        idx = torch.randperm(tx_symbols.size(0))[:sample_size]
        tx_symbols = tx_symbols[idx]

    diffs = tx_symbols.unsqueeze(1) - tx_symbols.unsqueeze(0)
    dists_squared = (diffs ** 2).sum(dim=-1) + epsilon  # add epsilon for numerical stability
    inv_dists = 1.0 / dists_squared
    loss = inv_dists.sum() - inv_dists.diagonal().sum()
    return loss / (tx_symbols.size(0) ** 2)



#--------------------------------------------------------------------------------------------------
# Define the basic FNN model
class FiberOpticFNN0(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(FiberOpticFNN0, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),  # Dropout for regularization
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Define the deeper model
class FiberOpticFNN1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(FiberOpticFNN1, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Define the wider model
class FiberOpticFNN2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(FiberOpticFNN2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

# Define the dynamic model
class FiberOpticFNN3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(FiberOpticFNN3, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, int(hidden_dim * 0.75)),
            nn.BatchNorm1d(int(hidden_dim * 0.75)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * 0.75), int(hidden_dim * 0.5)),
            nn.BatchNorm1d(int(hidden_dim * 0.5)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * 0.5), output_dim)
        )
    def forward(self, x):
        return self.fc(x)

# Define the noise-resilient model
class FiberOpticFNN4(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(FiberOpticFNN4, self).__init__()

        # Initial feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Noise-focused branch (captures small deviations)
        self.noise_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        # Residual connection for refined outputs
        self.residual = nn.Linear(hidden_dim, hidden_dim)

        # Final layer combining noise and refined features
        self.combined = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)

        # Process noise-sensitive features
        noise_features = self.noise_branch(features)

        # Add residual connection
        refined_features = features + self.residual(features)

        # Combine noise-sensitive and refined features
        combined_input = torch.cat((refined_features, noise_features), dim=1)

        # Final output
        output = self.combined(combined_input)
        return output

#--------------------------------------------------------------------------------------------------------------------

class TransmitterReceiverModel(nn.Module):
    def __init__(self, input_size, hidden_size, channel_size, channel_func, window_length):
        super().__init__()
        self.input_size = input_size
        self.window_length = window_length
        self.channel_size = channel_size
        self.channel_function = channel_func

        # Transmitter (4-layer MLP)
        self.transmitter = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, channel_size),  # final I/Q output
            nn.Sigmoid()
        )

        # Receiver (5-layer MLP)
        self.receiver = nn.Sequential(
            nn.Linear(channel_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, input_size)  # output logits
        )

        self.transmitter_scale = nn.Parameter(torch.tensor(1.0))  # learnable scalar

    def forward(self, x):
        num_windows = x.shape[0] // self.window_length
        x_windows = x.view(num_windows, self.window_length, self.input_size)
        x_flat = x_windows.view(-1, self.input_size)


        # Transmitter
        tx_symbols = normalize_symbols(self.transmitter(x_flat) * self.transmitter_scale)

        # Reshape to windows
        tx_windows = tx_symbols.view(num_windows, self.window_length, self.channel_size)
        tx_windows_flat = tx_windows.view(num_windows, -1)

        # Channel
        channel_out = self.channel_function(tx_windows_flat)

        # Receiver
        rx_output = self.receiver(channel_out)
        return rx_output

    def train_model(self, x_train, y_train, x_val, y_val, x_test, y_test, learning_rate, patience, epochs=10):
        """
        Args:
            x_train: Tensor of shape (total_length, input_size)
            y_train: Tensor of shape (num_windows,) with the target symbol (class index) for each window.
            x_val:
            y_val:
            x_test:
            y_test:
            learning_rate:
            patience:
            epochs:
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        train_loss = []
        val_loss = []
        best_val_loss = float('inf')
        counter = 0
        initial_lambda_spread = 0.005  # or higher if needed

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            # ----------- Forward Pass for Cross Entropy -----------
            y_pred = self.forward(x_train)
            loss_ce = criterion(y_pred, y_train)

            # ----------- Compute Spread Loss -----------

            lambda_spread = max(initial_lambda_spread * (0.95 ** epoch), 1e-4)

            x_flat = x_train.view(-1, self.input_size)
            tx_symbols = normalize_symbols(self.transmitter(x_flat) * self.transmitter_scale)

            num_windows = x_train.shape[0] // self.window_length
            tx_windows = tx_symbols.view(num_windows, self.window_length, self.channel_size)
            tx_windows_flat = tx_windows.view(num_windows, -1)

            with torch.no_grad():
                ch_out = self.channel_function(tx_windows_flat)

            loss_spread = compute_spread_loss(ch_out)

            # ----------- Total Loss -----------
            total_loss = loss_ce + lambda_spread * loss_spread
            total_loss.backward()
            optimizer.step()

            train_loss.append(total_loss.item())

            # Validation phase
            self.eval()
            with torch.no_grad():
                y_pred_val = self.forward(x_val)
                current_val_loss = criterion(y_pred_val, y_val).item()
                val_loss.append(current_val_loss)

                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_weights = {k: v.clone().detach() for k, v in self.state_dict().items()}

                    counter = 0
                else:
                    counter += 1

            if epoch % 10 == 0:
                print(f"Epoch {epoch} - Train Loss: {total_loss.item():.4f}, Val Loss: {current_val_loss:.4f}, loss spread: {loss_spread.item():.4f}")

            if counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break

        # Testing phase
        self.load_state_dict(best_weights)
        y_test_pred = torch.argmax(self.forward(x_test), dim=1)

        return train_loss, val_loss, y_test, y_test_pred, best_weights




# def Objective(trial, alphabet_size, total_length, channel):
def normalize_symbols(tx_symbols, target_norm=1.0):
    norm = tx_symbols.norm(dim=1, keepdim=True) + 1e-8
    return tx_symbols * (target_norm / norm)

# Parameters for this example:
alphabet_size = 8   # One-hot vector dimension; using 4096 symbols.
hidden_size = 256
channel_size = 2        # I/Q components.
window_size = 512       # Each window has 1024 symbols.
total_length = 40960     # Total number of symbols (4096/1024 = 4 windows).

weights, params = loader(
    r"C:\Users\alexa\PycharmProjects\CapstoneProject\.venv\training\Constellation\8QAM\Deeper\model_weights.pth",
    r"C:\Users\alexa\PycharmProjects\CapstoneProject\.venv\training\Constellation\8QAM\Deeper\best_params.json"
)

# -----------------------------------------------------------------------------
# Set up the fiber channel.
# The channel expects a flattened input of size 2 * window_length = 2 * 1024 = 2048.
channel = FiberOpticFNN1(2 * window_size, params["hidden_dim"], channel_size, params["dropout"])
channel.load_state_dict(weights["model weights"], strict=False)

# Freeze the channel to prevent training
for param in channel.parameters():
    param.requires_grad = False

channel_func = lambda x: channel.forward(x)

# Instantiate the autoencoder model.
model = TransmitterReceiverModel(
    input_size=alphabet_size,
    hidden_size=hidden_size,
    channel_size=channel_size,
    channel_func=channel_func,
    window_length=window_size
)

# -----------------------------------------------------------------------------
train_input, train_target = generate_data(alphabet_size, total_length)
val_input, val_target = generate_data(alphabet_size, total_length//4)
test_input, test_target = generate_data(alphabet_size, total_length//4)

# Train the model and plot loss.
train_loss, val_loss, y_test, y_test_pred, best_weights = model.train_model(
    train_input,
    train_target,
    val_input,
    val_target,
    test_input,
    test_target,
    learning_rate=0.01,
    patience=25,
    epochs=250)

# Calculate SER
num_errors = (y_test_pred != y_test).sum().item()
ser = num_errors / y_test.size(0)
print(f"Symbol Error Rate (SER): {ser:.4f}")

#-----------------------------------------------------
# -------------------- CONSTELLATION VISUALIZATION AFTER TRAINING --------------------
num_windows = 10
num_symbols = num_windows * window_size
input_indices = torch.randint(0, alphabet_size, (num_symbols,))
input_onehot = torch.eye(alphabet_size)[input_indices]  # Shape: (num_symbols, alphabet_size)

model.eval()
channel.eval()

with torch.no_grad():
    # Pass through full transmitter
    tx_symbols = model.transmitter(input_onehot)  # Shape: (num_symbols, 2)

    # Reshape for channel input
    tx_windows = tx_symbols.view(num_windows, -1)  # Shape: (num_windows, window_size * 2)

    # Channel output (already frozen)
    channel_out = channel(tx_windows)  # Shape: (num_windows, 2)




def plot_constellations(transmitter, channel, alphabet_size, window_size):
    num_windows = 10
    num_symbols = num_windows * window_size
    input_indices = torch.randint(0, alphabet_size, (num_symbols,))
    input_onehot = torch.eye(alphabet_size)[input_indices]

    with torch.no_grad():
        tx_symbols = normalize_symbols(transmitter(input_onehot))
        tx_windows = tx_symbols.view(num_windows, -1)
        channel_out = channel(tx_windows)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(tx_symbols[:, 0].numpy(), tx_symbols[:, 1].numpy(), alpha=0.6, s=30, edgecolors='k')
    plt.title("Transmitter Output (Before Channel)")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid(True)
    plt.axis("equal")

    plt.subplot(1, 2, 2)
    plt.scatter(channel_out[:, 0].numpy(), channel_out[:, 1].numpy(), alpha=0.6, s=30, c='red', edgecolors='k')
    plt.title("Channel Output (After Channel)")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid(True)
    plt.axis("equal")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(tx_symbols[:, 0].numpy(), tx_symbols[:, 1].numpy(), alpha=0.6, s=30, label="Before Channel", edgecolors='k')
    plt.scatter(channel_out[:, 0].numpy(), channel_out[:, 1].numpy(), alpha=0.6, s=30, label="After Channel", c='red', edgecolors='k')
    plt.title("Constellation Comparison (Before vs After Channel)")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

plot_constellations(
    transmitter=model.transmitter,
    channel=channel,
    alphabet_size=alphabet_size,
    window_size=window_size
)
