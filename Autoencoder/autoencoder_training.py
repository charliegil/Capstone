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
    input = np.random.randint(0, alphabet_size-1, total_size)
    input_one_hot = np.eye(alphabet_size)[input]
    input_windows = input.reshape(-1, window_size)
    target_data = input_windows[:, window_size // 2]

    # Convert data to PyTorch tensors.
    input_tensor = torch.tensor(input_one_hot, dtype=torch.float32)
    target_tensor = torch.tensor(target_data, dtype=torch.long)

    return input_tensor, target_tensor

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
    def __init__(self, input_size, hidden_size, channel_size, channel_func, window_length, scaling_factor):
        """
        Args:
            input_size (int): Dimensionality of one-hot vectors (alphabet size).
            hidden_size (int): Hidden layer dimension.
            channel_size (int): Dimension of the transmitter output (e.g., 2 for I/Q).
            channel_func (callable): The fiber channel function. It expects an input of size (2 * window_length).
            window_length (int): Number of symbols per window.
        """
        super().__init__()
        self.input_size = input_size      # e.g., 4096 (alphabet size)
        self.window_length = window_length  # e.g., 1024
        self.channel_size = channel_size    # e.g., 2 (I/Q)
        self.scaling_factor = scaling_factor

        # Transmitter: processes one symbol at a time.
        self.transmitter_fc1 = nn.Linear(input_size, hidden_size)
        self.transmitter_fc2 = nn.Linear(hidden_size, channel_size)

        # Receiver: decodes the channel output back to logits over the alphabet.
        self.receiver_fc1 = nn.Linear(channel_size, hidden_size)
        self.receiver_fc2 = nn.Linear(hidden_size, input_size)

        self.channel_function = channel_func

    def forward(self, x):
        """
        First passes each symbol (one-hot encoded vectors) through the transmitter.
        Then arranges the transmitter outputs into windows, and passes each window through the channel.
        Then passes the channel output through the receiver.
        Args:
            x: Tensor of shape (total_length, input_size), where total_length is divisible by window_length.
        Returns:
            A tensor of shape (num_windows, input_size), where each row corresponds to the receiver's
            output for that window.
        """
        # Group input into non-overlapping windows of size window_length.
        num_windows = x.shape[0] // self.window_length #total_length / window_length
        # New shape: (num_windows, window_length, input_size)
        x_windows = x.view(num_windows, self.window_length, self.input_size)

        # flatten input to get an array of all the one hot encoded vectors
        x_flat = x_windows.view(-1, self.input_size)  # Shape: (num_windows * window_length, input_size)
        tx_hidden = torch.sigmoid(self.transmitter_fc1(x_flat)) # passing each symbols one at a time (first layer)
        # mapping each symbol to an (I,Q) pair (second layer)
        tx_symbols = self.transmitter_fc2(tx_hidden)     # Shape: (num_windows * window_length, channel_size)

        # Reshape back into windows: (num_windows, window_length, channel_size)
        tx_windows = tx_symbols.view(num_windows, self.window_length, self.channel_size)

        # Flatten each window to create a vector of size (window_length * channel_size).
        # With window_length=1024 and channel_size=2, this gives a vector of size 2048.
        tx_windows_flat = tx_windows.view(num_windows, -1) # [I1, Q1, I2, Q2, ...]

        # Pass each flattened window through the fiber channel.
        channel_out = self.channel_function(tx_windows_flat)  # Expected shape: (num_windows, channel_size)

        # Decode the channel output using the receiver.
        rx_hidden = torch.sigmoid(self.receiver_fc1(channel_out))
        rx_output = self.receiver_fc2(rx_hidden)  # Shape: (num_windows, input_size)
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

        for epoch in range(epochs):
            # Training phase
            self.train()
            optimizer.zero_grad()
            y_pred = self.forward(x_train)  # Shape: (num_windows, input_size)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            current_train_loss = loss.item()
            train_loss.append(current_train_loss)

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

            if epoch % 25 == 0:
                print(f"Epoch {epoch} - Train Loss: {current_train_loss:.4f}, Val Loss: {current_val_loss:.4f}")

            if counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break

        # Testing phase
        self.load_state_dict(best_weights)
        y_test_pred = torch.argmax(self.forward(x_test), dim=1)

        return train_loss, val_loss, y_test, y_test_pred, best_weights


# def Objective(trial, alphabet_size, total_length, channel):
#     best_val_loss = float('inf')


# Parameters for this example:
alphabet_size = 4   # One-hot vector dimension; using 4096 symbols.
hidden_size = 100
channel_size = 2        # I/Q components.
window_size = 512       # Each window has 1024 symbols.
total_length = 40960     # Total number of symbols (4096/1024 = 4 windows).

weights, params = loader(
    r"C:\Users\alexa\PycharmProjects\CapstoneProject\.venv\training\Constellation\QPSK\Wider\model_weights.pth",
    r"C:\Users\alexa\PycharmProjects\CapstoneProject\.venv\training\Constellation\QPSK\Wider\best_params.json"
)

# -----------------------------------------------------------------------------
# Set up the fiber channel.
# The channel expects a flattened input of size 2 * window_length = 2 * 1024 = 2048.
channel = FiberOpticFNN2(2 * window_size, params["hidden_dim"], channel_size, params["dropout"])
channel.load_state_dict(weights["model weights"], strict=False)
#channel_func = lambda x: channel.forward(x)
def identity_channel(x):
    # Input is [num_windows, window_length * channel_size]
    # Return [num_windows, channel_size] by averaging across the window
    return x.view(x.shape[0], -1, 2).mean(dim=1)


channel_func = identity_channel

# Instantiate the autoencoder model.
model = TransmitterReceiverModel(
    input_size=alphabet_size,
    hidden_size=hidden_size,
    channel_size=channel_size,
    channel_func=channel_func,
    window_length=window_size,
    scaling_factor=params["scaling_factor"]
)

# -----------------------------------------------------------------------------
train_input, train_target = generate_data(alphabet_size, total_length)
val_input, val_target = generate_data(alphabet_size, total_length//4)
test_input, test_target = generate_data(alphabet_size, total_length//4)

# -----------------------------------------------------------------------------
# Step 1: Generate some symbols
num_symbols = 100
input_indices = torch.randint(0, alphabet_size, (num_symbols,))
input_onehot = torch.eye(alphabet_size)[input_indices]  # Shape: (num_symbols, alphabet_size)

# Step 2: Pass through transmitter manually
model.eval()
with torch.no_grad():
    tx_hidden = torch.sigmoid(model.transmitter_fc1(input_onehot))
    tx_symbols = model.transmitter_fc2(tx_hidden)  # Shape: (num_symbols, channel_size)

# Step 3: Group into a fake window and pass through the channel
tx_window = tx_symbols.view(1, -1)  # Shape: (1, num_symbols * channel_size)
channel_out = model.channel_function(tx_window) / model.scaling_factor  # Shape: (1, channel_size)

# Step 4 (Optional): Visualize

plt.figure(figsize=(6, 6))
plt.scatter(tx_symbols[:, 0].numpy(), tx_symbols[:, 1].numpy(), label="Transmitter Output")
plt.title("I/Q Constellation Before Channel")
plt.xlabel("I")
plt.ylabel("Q")
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()

# Train the model and plot loss.
train_loss, val_loss, y_test, y_test_pred, best_weights = model.train_model(
    train_input,
    train_target,
    val_input,
    val_target,
    test_input,
    test_target,
    learning_rate=0.01,
    patience=75,
    epochs=1000)

# Calculate SER
num_errors = (y_test_pred != y_test).sum().item()
ser = num_errors / y_test.size(0)
print(f"Symbol Error Rate (SER): {ser:.4f}")