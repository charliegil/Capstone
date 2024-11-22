import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Load data
input_data = pd.read_csv(".venv/OOK_input_data.txt", delim_whitespace=True, header=None, names=["Time", "Amplitude"]).to_numpy()
output_data = pd.read_csv(".venv/OOK_output_data.txt", delim_whitespace=True, header=None, names=["Time", "Amplitude"]).to_numpy()

#Normalize data using min maxing
input_data[:, 1] = (input_data[:, 1] - input_data[:, 1].min()) / (input_data[:, 1].max() - input_data[:, 1].min())
output_data[:, 1] = (output_data[:, 1] - output_data[:, 1].min()) / (output_data[:, 1].max() - output_data[:, 1].min())

data = np.column_stack((input_data[:, 0], input_data[:, 1], output_data[:, 1]))

def create_windows(data, window_size, step_size):
    num_windows = (len(data) - window_size) // step_size + 1
    windows = np.array([
        data[i:i + window_size]  # Extract rows for each window
        for i in range(0, num_windows * step_size, step_size)
    ])
    return windows

window_size = 500
step_size = 1
windows = create_windows(data, window_size, step_size)

# Split the windows into inputs (X) and targets (y)
X = windows[:, :, [0, 1]]  # Time and input amplitude
y = windows[:, :, [0, 2]]  # Time and output amplitude

# Print the shapes
#(f"X Shape: {X.shape}")  # Expected: (num_windows, window_size, 2)
#print(f"y Shape: {y.shape}")  # Expected: (num_windows, window_size, 2)

# Define sizes for training, validation, and testing
train_size = int(0.7 * len(X))  # 70% for training
val_size = int(0.15 * len(X))   # 15% for validation
test_size = len(X) - train_size - val_size  # Remaining 15% for testing

#split data
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Print shapes to confirm
print(f"X_train Shape: {X_train.shape}, y_train Shape: {y_train.shape}")
print(f"X_val Shape: {X_val.shape}, y_val Shape: {y_val.shape}")
print(f"X_test Shape: {X_test.shape}, y_test Shape: {y_test.shape}")

# Convert the training set to PyTorch tensors with double precision
X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
y_train_tensor = torch.tensor(y_train, dtype=torch.float64)

# Convert the validation set to PyTorch tensors with double precision
X_val_tensor = torch.tensor(X_val, dtype=torch.float64)
y_val_tensor = torch.tensor(y_val, dtype=torch.float64)

# Convert the testing set to PyTorch tensors with double precision
X_test_tensor = torch.tensor(X_test, dtype=torch.float64)
y_test_tensor = torch.tensor(y_test, dtype=torch.float64)

# Print shapes to confirm
print(f"X_train_tensor Shape: {X_train_tensor.shape}, y_train_tensor Shape: {y_train_tensor.shape}")
print(f"X_val_tensor Shape: {X_val_tensor.shape}, y_val_tensor Shape: {y_val_tensor.shape}")
print(f"X_test_tensor Shape: {X_test_tensor.shape}, y_test_tensor Shape: {y_test_tensor.shape}")

# Create TensorDatasets with retained time
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Define batch size
batch_size = 64

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # No shuffling for time-series
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class FiberOpticLSTM_Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(FiberOpticLSTM_Nonlinear, self).__init__()
        self.hidden_dim = hidden_dim

        # Nonlinear preprocessing layer
        self.nonlinear_preprocess = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),  # Expand features
            nn.ReLU(),  # Nonlinear activation
            nn.Linear(input_dim * 2, input_dim)  # Back to original dimension
        )

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Nonlinear feature preprocessing
        x = self.nonlinear_preprocess(x)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # Ignore hidden and cell states

        # Fully connected layer for each time step
        predictions = self.fc(lstm_out)

        return predictions
# Instantiate the model
input_dim = 2  # Time and input amplitude
hidden_dim = 64  # Number of LSTM units
num_layers = 2  # Number of LSTM layers
output_dim = 2  # Time and output amplitude

model = FiberOpticLSTM_Nonlinear(input_dim, hidden_dim, num_layers, output_dim)
model = model.double()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping configuration
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, validation_loss):
        if validation_loss < self.best_loss - self.delta:
            self.best_loss = validation_loss
            self.counter = 0
            return False  # Don't stop
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        return False

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10):
    train_losses = []
    val_losses = []

    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

        if early_stopping.step(val_losses[-1]):
            print("Early stopping triggered")
            break

    return train_losses, val_losses

# Train the model
batch_size = 64
num_epochs = 100
train_losses, val_losses = train_model(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=num_epochs,
    patience=10
)

# Plot learning curves
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
