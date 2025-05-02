import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
input_data = pd.read_csv(".venv/OOK_input_data.txt", delim_whitespace=True, header=None,names=["Time", "Amplitude"]).to_numpy()
output_data = pd.read_csv(".venv/OOK_output_data.txt", delim_whitespace=True, header=None,names=["Time", "Amplitude"]).to_numpy()

# Validate raw data
print("Raw Input Data (First 5 rows):")
print(input_data[:5])
print("Raw Output Data (First 5 rows):")
print(output_data[:5])

# Standardize input and output data
input_mean, input_std = input_data[:, 1].mean(), input_data[:, 1].std()
output_mean, output_std = output_data[:, 1].mean(), output_data[:, 1].std()

input_data[:, 1] = (input_data[:, 1] - input_mean) / input_std
output_data[:, 1] = (output_data[:, 1] - output_mean) / output_std

# Validate Standardized data
print("\nStandardized Input Data (First 5 rows):")
print(input_data[:5])
print("Standardized Output Data (First 5 rows):")
print(output_data[:5])

data = np.column_stack((input_data[:, 0], input_data[:, 1], output_data[:, 1]))

# Validate combined data
print("\nCombined Data (First 5 rows):")
print(data[:5])


# Sliding window function
def create_windows(data, window_size, step_size):
    num_windows = (len(data) - window_size) // step_size + 1
    windows = np.array([
        data[i:i + window_size]  # Extract rows for each window
        for i in range(0, num_windows * step_size, step_size)
    ])
    return windows


# Define window size and step size
window_size = 750
step_size = 1
windows = create_windows(data, window_size, step_size)

# Validate sliding windows
print("\nSliding Windows (Shape):", windows.shape)
print("First Sliding Window (First 5 rows):")
print(windows[0][:5])

# Flatten the input windows for FNN
X = windows[:, :, 1].reshape(windows.shape[0], -1)  # Amplitude only
y = windows[:, -1, 2]  # Output amplitude for the last time step
time_index = windows[:, :, 0]  # Time values retained for indexing

# Validate flattened input and output
print("\nFlattened Input X (Shape):", X.shape)
print("First Flattened Input X (First 5 values):")
print(X[0][:5])
print("\nOutput y (Shape):", y.shape)
print("First 5 Output Values y:")
print(y[:5])
print("\nTime Index (Shape):", time_index.shape)
print("First Time Index (First 5 rows):")
print(time_index[:5])

# Split data into training, validation, and testing sets
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
test_size = len(X) - train_size - val_size

X_train, y_train, time_train = X[:train_size], y[:train_size], time_index[:train_size]
X_val, y_val, time_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size], time_index[train_size:train_size + val_size]
X_test, y_test, time_test = X[train_size + val_size:], y[train_size + val_size:], time_index[train_size + val_size:]

# Validate splits
print("\nTraining Set Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("\nValidation Set Shapes:")
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print("\nTest Set Shapes:")
print("X_test:", X_test.shape, "y_test:", y_test.shape)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Validate tensor shapes
print("\nTensor Shapes:")
print("X_train_tensor:", X_train_tensor.shape, "y_train_tensor:", y_train_tensor.shape)
print("X_val_tensor:", X_val_tensor.shape, "y_val_tensor:", y_val_tensor.shape)
print("X_test_tensor:", X_test_tensor.shape, "y_test_tensor:", y_test_tensor.shape)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Validate DataLoader
print("\nDataLoader Validation:")
for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
    print(f"Batch {batch_idx + 1} - X_batch Shape: {X_batch.shape}, y_batch Shape: {y_batch.shape}")
    break  # Print only the first batch


# Define the FNN model
class FiberOpticFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FiberOpticFNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout for regularization
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


# Instantiate the model
input_dim = X_train.shape[1]  # Flattened window size
hidden_dim = 128  # Number of hidden units
output_dim = 1  # Output amplitude

model = FiberOpticFNN(input_dim, hidden_dim, output_dim)
model = model.float()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)


# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10):
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience_counter = 0

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

        # Early stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return train_losses, val_losses


# Train the model
train_losses, val_losses = train_model(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=100,
    patience=10
)

model.eval()  # Set the model to evaluation mode
test_predictions = []
with torch.no_grad():
    for X_batch in test_loader:
        predictions = model(X_batch[0])  # X_batch[0] is the input tensor
        test_predictions.extend(predictions.numpy())

# Convert predictions and ground truth to numpy
test_predictions = np.array(test_predictions).flatten()
y_test_np = y_test_tensor.numpy().flatten()

# Calculate accuracy metrics
mse = mean_squared_error(y_test_np, test_predictions)
mae = mean_absolute_error(y_test_np, test_predictions)

# Print metrics
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")

# Display predictions with time indices
for i, prediction in enumerate(test_predictions[:10]):  # Display first 10 predictions
    print(f"Time: {time_test[i, -1]}, Predicted Amplitude: {prediction:.4f}, True Amplitude: {y_test_np[i]:.4f}")

# Plot training and validation loss curves
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
