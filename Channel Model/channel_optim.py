import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import mpld3

input_paths = [".venv/OOK_input_data.txt", ".venv/PAM4_input_data.txt", ".venv/QAM_input_data.txt"]
output_paths = [".venv/OOK_output_data.txt", ".venv/PAM4_output_data.txt", ".venv/QAM_output_data.txt"]

#initialize lists
input_data = []
output_data = []

for i in range(len(input_paths)):

    input_data_temp = pd.read_csv(input_paths[i], sep = r'\s+', header = None, names = ["Time", "Amplitude"]).to_numpy()
    output_data_temp = pd.read_csv(output_paths[i], sep = r'\s+', header = None, names = ["Time", "Amplitude"]).to_numpy()

    # Align data sizes by truncating to the minimum length
    if len(input_data_temp) != len(output_data_temp):
        min_length = min(len(input_data_temp), len(output_data_temp))
        input_data_temp = input_data_temp[:min_length]
        output_data_temp = output_data_temp[:min_length]

    # Replace NaN values with the mean of the column
    if np.isnan(input_data_temp[:, 1]).any():# Check for NaN values
        input_mean = np.nanmean(input_data_temp[:, 1])  # Mean ignoring NaNs
        input_data_temp[np.isnan(input_data_temp[:, 1]), 1] = input_mean

    if np.isnan(output_data_temp[:, 1]).any():  # Check for NaN values
        output_mean = np.nanmean(output_data_temp[:, 1])  # Mean ignoring NaNs
        output_data_temp[np.isnan(output_data_temp[:, 1]), 1] = output_mean

    #append arrays to list
    input_data.append(input_data_temp)
    output_data.append(output_data_temp)

#convert lists to numpy arrays
input_data = np.vstack(input_data)
output_data = np.vstack(output_data)

print(f"Input data shape: {input_data.shape}")
print(f"Output data shape: {output_data.shape}")

# Validate raw data
print(f"Number of datasets: {len(input_data)}")
print(f"First dataset input shape: {input_data[0].shape}")
print(f"First dataset output shape: {output_data[0].shape}")

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
window_size = 1024
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



# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=15):
    train_losses = []
    val_losses = []
    best_weights = None

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
            best_weights = {"model weights": model.state_dict()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return train_losses, val_losses, best_weights



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
class FibreOpticFNN4(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(FibreOpticFNN4, self).__init__()

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

# Define the Residual Connections model
class FibreOpticFNN5(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FibreOpticFNN5, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = x + self.hidden_layer(x)  # Residual connection
        return self.output_layer(x)

# Basic
def objective0(trial):

    best_val_loss = float("inf")

    # Hyperparameters to tune
    hidden_dim = trial.suggest_int("hidden dim", 128, 320, step= 16)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log = True)
    weight_decay = trial.suggest_float("weight decay", 1e-6, 1e-3, log =True)
    dropout_rate = trial.suggest_float("dropout rate", 0.1, 0.5)

    #instatiate the model
    model = FiberOpticFNN0(X_train.shape[1], hidden_dim, 1, dropout_rate)
    model = model.float()

    criterion = nn.MSELoss() #define loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) #define optimizer

    # Train the model
    train_losses, val_losses, best_weights = train_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=100,
        patience=15
    )

    if min(val_losses) < best_val_loss:
        best_val_loss = min(val_losses)
        best_params = {"model weights": best_weights,
                       "hidden_dim": hidden_dim,
                       "lr": lr,
                       "weight_decay": weight_decay,
                       "dropout_rate": dropout_rate,
                       "train_losses": train_losses,
                       "val_losses": val_losses
                       }
        torch.save(best_params, ".venv/basic/best_params.pth")

    return best_val_loss

#deeper
def objective1(trial):

    best_val_loss = float("inf")

    # Hyperparameters to tune
    hidden_dim = trial.suggest_int("hidden dim", 128, 320, step= 16)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log = True)
    weight_decay = trial.suggest_float("weight decay", 1e-6, 1e-3, log =True)
    dropout_rate = trial.suggest_float("dropout rate", 0.1, 0.5)

    #instatiate the model
    model = FiberOpticFNN1(X_train.shape[1], hidden_dim, 1, dropout_rate)
    model = model.float()

    criterion = nn.MSELoss() #define loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) #define optimizer

    # Train the model
    train_losses, val_losses, best_weights = train_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=100,
        patience=15
    )

    if min(val_losses) < best_val_loss:
        best_val_loss = min(val_losses)
        best_params = {"model weights": best_weights,
                       "hidden_dim": hidden_dim,
                       "lr": lr,
                       "weight_decay": weight_decay,
                       "dropout_rate": dropout_rate,
                       "train_losses": train_losses,
                       "val_losses": val_losses
                       }

        torch.save(best_params, ".venv/deeper/best_params.pth")

    return best_val_loss

# Wider
def objective2(trial):
    best_val_loss = float("inf")

    # Hyperparameters to tune
    hidden_dim = trial.suggest_int("hidden dim", 128, 320, step= 16)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log = True)
    weight_decay = trial.suggest_float("weight decay", 1e-6, 1e-3, log =True)
    dropout_rate = trial.suggest_float("dropout rate", 0.1, 0.5)

    #instatiate the model
    model = FiberOpticFNN2(X_train.shape[1], hidden_dim, 1, dropout_rate)
    model = model.float()

    criterion = nn.MSELoss() #define loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) #define optimizer

    # Train the model
    train_losses, val_losses, best_weights = train_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=100,
        patience=15
    )

    if min(val_losses) < best_val_loss:
        best_val_loss = min(val_losses)
        best_params = {"model weights": best_weights,
                       "hidden_dim": hidden_dim,
                       "lr": lr,
                       "weight_decay": weight_decay,
                       "dropout_rate": dropout_rate,
                       "train_losses": train_losses,
                       "val_losses": val_losses
                       }

        torch.save(best_params, ".venv/wider/best_params.pth")

    return best_val_loss

# Dynamic
def objective3(trial):
    best_val_loss = float("inf")

    # Hyperparameters to tune
    hidden_dim = trial.suggest_int("hidden dim", 128, 320, step= 16)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log = True)
    weight_decay = trial.suggest_float("weight decay", 1e-6, 1e-3, log =True)
    dropout_rate = trial.suggest_float("dropout rate", 0.1, 0.5)

    #instatiate the model
    model = FiberOpticFNN3(X_train.shape[1], hidden_dim, 1, dropout_rate)
    model = model.float()

    criterion = nn.MSELoss() #define loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) #define optimizer

    # Train the model
    train_losses, val_losses, best_weights = train_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=100,
        patience=15
    )

    if min(val_losses) < best_val_loss:
        best_val_loss = min(val_losses)
        best_params = {"model weights": best_weights,
                       "hidden_dim": hidden_dim,
                       "lr": lr,
                       "weight_decay": weight_decay,
                       "dropout_rate": dropout_rate,
                       "train_losses": train_losses,
                       "val_losses": val_losses
                       }

        torch.save(best_params, ".venv/dynamic/best_params.pth")

    return best_val_loss

# noise resilient
def objective4(trial):
    best_val_loss = float("inf")

    # Hyperparameters to tune
    hidden_dim = trial.suggest_int("hidden dim", 128, 320, step= 16)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log = True)
    weight_decay = trial.suggest_float("weight decay", 1e-6, 1e-3, log =True)
    dropout_rate = trial.suggest_float("dropout rate", 0.1, 0.5)

    #instatiate the model
    model = FibreOpticFNN4(X_train.shape[1], hidden_dim, 1, dropout_rate)
    model = model.float()

    criterion = nn.MSELoss() #define loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) #define optimizer

    # Train the model
    train_losses, val_losses, best_weights = train_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=100,
        patience=15
    )

    if min(val_losses) < best_val_loss:
        best_val_loss = min(val_losses)
        best_params = {"model weights": best_weights,
                       "hidden_dim": hidden_dim,
                       "lr": lr,
                       "weight_decay": weight_decay,
                       "dropout_rate": dropout_rate,
                       "train_losses": train_losses,
                       "val_losses": val_losses
                       }
        torch.save(best_params, ".venv/NR/best_params.pth")

    return best_val_loss

#Residual Connections
def objective5(trial):
    best_val_loss = float("inf")

    # Hyperparameters to tune
    hidden_dim = trial.suggest_int("hidden dim", 128, 320, step= 16)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log = True)
    weight_decay = trial.suggest_float("weight decay", 1e-6, 1e-3, log =True)
    #dropout_rate = trial.suggest_float("dropout rate", 0.1, 0.5)

    #instatiate the model
    model = FibreOpticFNN5(X_train.shape[1], hidden_dim, 1)
    model = model.float()

    criterion = nn.MSELoss() #define loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) #define optimizer

    # Train the model
    train_losses, val_losses, best_weights = train_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=100,
        patience=15
    )
    if min(val_losses) < best_val_loss:
        best_val_loss = min(val_losses)
        best_params = {"model weights": best_weights,
                       "hidden_dim": hidden_dim,
                       "lr": lr,
                       "weight_decay": weight_decay,
                       #"dropout_rate": dropout_rate,
                       "train_losses": train_losses,
                       "val_losses": val_losses
                       }
        torch.save(best_params, ".venv/residual/best_params.pth")

    return best_val_loss



# Create a studies for hyperparameter optimization
study0 = optuna.create_study(direction="minimize")  # Minimize the validation loss
study0.optimize(objective0, n_trials=100)   #Run 100 trials

study1 = optuna.create_study(direction="minimize")
study1.optimize(objective1, n_trials=100)

study2 = optuna.create_study(direction="minimize")
study2.optimize(objective2, n_trials=100)

study3 = optuna.create_study(direction="minimize")
study3.optimize(objective3, n_trials=100)

study4 = optuna.create_study(direction="minimize")
study4.optimize(objective4, n_trials=100)

study5 = optuna.create_study(direction="minimize")
study5.optimize(objective5, n_trials=100)

#######################################################################################################################
#Evaluate the first model
params = torch.load(".venv/basic/best_params.pth")
model = FiberOpticFNN0(X_test.shape[1], params["hidden_dim"], 1, params["dropout_rate"])
model.load_state_dict(params["model_weights"])
model.eval()

test_predictions = []
with torch.no_grad():
    for X_batch in test_loader:
        predictions = model(X_batch[0])  # X_batch[0] is the input tensor
        test_predictions.extend(predictions.numpy())

# Convert predictions and ground truth to numpy
test_predictions = np.array(test_predictions).flatten()
y_test_np = y_test_tensor.numpy().flatten()

# Denormalize the predictions and actual values
test_predictions = test_predictions * output_std + output_mean
y_test_np = y_test_np * output_std + output_mean

r2 = r2_score(y_test_np, test_predictions)
mae = mean_absolute_error(y_test_np, test_predictions)
mse = mean_squared_error(y_test_np, test_predictions)

# Prepare the output string
metrics_text = f"""
Evaluation Metrics:
===================
R² Score: {r2:.4f}
Mean Absolute Error (MAE): {mae:.4f}
Mean Squared Error (MSE): {mse:.4f}
"""

# Define the output file path
metrics_file_path = ".venv/basic/evaluation_metrics.txt"

# Save to a text file
with open(metrics_file_path, "w") as file:
    file.write(metrics_text)

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test_np, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Predicted vs Actual Denormalized Values')
plt.legend()
mpld3.save_html(plt.gcf(), ".venv/basic/predicted_vs_actual.html")

train_losses = params["train_losses"]
val_losses = params["val_losses"]

# plot the training and validation loss curves
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True)  # Add grid for better readability
plt.tight_layout()
mpld3.save_html(plt.gcf(), ".venv/basic/loss_curves.html")


#######################################################################################################################
#Evaluate the second model
params = torch.load(".venv/deeper/best_params.pth")
model = FiberOpticFNN1(X_test.shape[1], params["hidden_dim"], 1, params["dropout_rate"])
model.load_state_dict(params["model_weights"])
model.eval()

test_predictions = []
with torch.no_grad():
    for X_batch in test_loader:
        predictions = model(X_batch[0])  # X_batch[0] is the input tensor
        test_predictions.extend(predictions.numpy())

# Convert predictions and ground truth to numpy
test_predictions = np.array(test_predictions).flatten()
y_test_np = y_test_tensor.numpy().flatten()

# Denormalize the predictions and actual values
test_predictions = test_predictions * output_std + output_mean
y_test_np = y_test_np * output_std + output_mean

r2 = r2_score(y_test_np, test_predictions)
mae = mean_absolute_error(y_test_np, test_predictions)
mse = mean_squared_error(y_test_np, test_predictions)

# Prepare the output string
metrics_text = f"""
Evaluation Metrics:
===================
R² Score: {r2:.4f}
Mean Absolute Error (MAE): {mae:.4f}
Mean Squared Error (MSE): {mse:.4f}
"""

# Define the output file path
metrics_file_path = ".venv/deeper/evaluation_metrics.txt"

# Save to a text file
with open(metrics_file_path, "w") as file:
    file.write(metrics_text)

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test_np, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Predicted vs Actual Denormalized Values')
plt.legend()
mpld3.save_html(plt.gcf(), ".venv/deeper/predicted_vs_actual.html")

train_losses = params["train_losses"]
val_losses = params["val_losses"]

# plot the training and validation loss curves
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True)  # Add grid for better readability
plt.tight_layout()
mpld3.save_html(plt.gcf(), ".venv/deeper/loss_curves.html")


#######################################################################################################################
#Evalute the third model
params = torch.load(".venv/wider/best_params.pth")
model = FiberOpticFNN2(X_test.shape[1], params["hidden_dim"], 1, params["dropout_rate"])
model.load_state_dict(params["model_weights"])
model.eval()

test_predictions = []
with torch.no_grad():
    for X_batch in test_loader:
        predictions = model(X_batch[0])  # X_batch[0] is the input tensor
        test_predictions.extend(predictions.numpy())

# Convert predictions and ground truth to numpy
test_predictions = np.array(test_predictions).flatten()
y_test_np = y_test_tensor.numpy().flatten()

# Denormalize the predictions and actual values
test_predictions = test_predictions * output_std + output_mean
y_test_np = y_test_np * output_std + output_mean

r2 = r2_score(y_test_np, test_predictions)
mae = mean_absolute_error(y_test_np, test_predictions)
mse = mean_squared_error(y_test_np, test_predictions)

# Prepare the output string
metrics_text = f"""
Evaluation Metrics:
===================
R² Score: {r2:.4f}
Mean Absolute Error (MAE): {mae:.4f}
Mean Squared Error (MSE): {mse:.4f}
"""

# Define the output file path
metrics_file_path = ".venv/wider/evaluation_metrics.txt"

# Save to a text file
with open(metrics_file_path, "w") as file:
    file.write(metrics_text)

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test_np, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Predicted vs Actual Denormalized Values')
plt.legend()
mpld3.save_html(plt.gcf(), ".venv/wider/predicted_vs_actual.html")

train_losses = params["train_losses"]
val_losses = params["val_losses"]

# plot the training and validation loss curves
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True)  # Add grid for better readability
plt.tight_layout()
mpld3.save_html(plt.gcf(), ".venv/wider/loss_curves.html")


#######################################################################################################################
#Evaluate the fourth model
params = torch.load(".venv/dynamic/best_params.pth")
model = FiberOpticFNN3(X_test.shape[1], params["hidden_dim"], 1, params["dropout_rate"])
model.load_state_dict(params["model_weights"])
model.eval()

test_predictions = []
with torch.no_grad():
    for X_batch in test_loader:
        predictions = model(X_batch[0])  # X_batch[0] is the input tensor
        test_predictions.extend(predictions.numpy())

# Convert predictions and ground truth to numpy
test_predictions = np.array(test_predictions).flatten()
y_test_np = y_test_tensor.numpy().flatten()

# Denormalize the predictions and actual values
test_predictions = test_predictions * output_std + output_mean
y_test_np = y_test_np * output_std + output_mean

r2 = r2_score(y_test_np, test_predictions)
mae = mean_absolute_error(y_test_np, test_predictions)
mse = mean_squared_error(y_test_np, test_predictions)

# Prepare the output string
metrics_text = f"""
Evaluation Metrics:
===================
R² Score: {r2:.4f}
Mean Absolute Error (MAE): {mae:.4f}
Mean Squared Error (MSE): {mse:.4f}
"""

# Define the output file path
metrics_file_path = ".venv/dynamic/evaluation_metrics.txt"

# Save to a text file
with open(metrics_file_path, "w") as file:
    file.write(metrics_text)

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test_np, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Predicted vs Actual Denormalized Values')
plt.legend()
mpld3.save_html(plt.gcf(), ".venv/dynamic/predicted_vs_actual.html")

train_losses = params["train_losses"]
val_losses = params["val_losses"]

# plot the training and validation loss curves
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True)  # Add grid for better readability
plt.tight_layout()
mpld3.save_html(plt.gcf(), ".venv/dynamic/loss_curves.html")


#######################################################################################################################
#Evaluate the fifth model
params = torch.load(".venv/NR/best_params.pth")
model = FibreOpticFNN4(X_test.shape[1], params["hidden_dim"], 1, params["dropout_rate"])
model.load_state_dict(params["model_weights"])
model.eval()

test_predictions = []
with torch.no_grad():
    for X_batch in test_loader:
        predictions = model(X_batch[0])  # X_batch[0] is the input tensor
        test_predictions.extend(predictions.numpy())

# Convert predictions and ground truth to numpy
test_predictions = np.array(test_predictions).flatten()
y_test_np = y_test_tensor.numpy().flatten()

# Denormalize the predictions and actual values
test_predictions = test_predictions * output_std + output_mean
y_test_np = y_test_np * output_std + output_mean

r2 = r2_score(y_test_np, test_predictions)
mae = mean_absolute_error(y_test_np, test_predictions)
mse = mean_squared_error(y_test_np, test_predictions)

# Prepare the output string
metrics_text = f"""
Evaluation Metrics:
===================
R² Score: {r2:.4f}
Mean Absolute Error (MAE): {mae:.4f}
Mean Squared Error (MSE): {mse:.4f}
"""

# Define the output file path
metrics_file_path = ".venv/NR/evaluation_metrics.txt"

# Save to a text file
with open(metrics_file_path, "w") as file:
    file.write(metrics_text)

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test_np, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Predicted vs Actual Denormalized Values')
plt.legend()
mpld3.save_html(plt.gcf(), ".venv/NR/predicted_vs_actual.html")

train_losses = params["train_losses"]
val_losses = params["val_losses"]

# plot the training and validation loss curves
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True)  # Add grid for better readability
plt.tight_layout()
mpld3.save_html(plt.gcf(), ".venv/NR/loss_curves.html")

#######################################################################################################################
#Evaluate the sixth model
params = torch.load(".venv/residual/best_params.pth")
model = FibreOpticFNN5(X_test.shape[1], params["hidden_dim"], 1)
model.load_state_dict(params["model_weights"])
model.eval()

test_predictions = []
with torch.no_grad():
    for X_batch in test_loader:
        predictions = model(X_batch[0])  # X_batch[0] is the input tensor
        test_predictions.extend(predictions.numpy())

# Convert predictions and ground truth to numpy
test_predictions = np.array(test_predictions).flatten()
y_test_np = y_test_tensor.numpy().flatten()

# Denormalize the predictions and actual values
test_predictions = test_predictions * output_std + output_mean
y_test_np = y_test_np * output_std + output_mean

r2 = r2_score(y_test_np, test_predictions)
mae = mean_absolute_error(y_test_np, test_predictions)
mse = mean_squared_error(y_test_np, test_predictions)

# Prepare the output string
metrics_text = f"""
Evaluation Metrics:
===================
R² Score: {r2:.4f}
Mean Absolute Error (MAE): {mae:.4f}
Mean Squared Error (MSE): {mse:.4f}
"""

# Define the output file path
metrics_file_path = ".venv/residual/evaluation_metrics.txt"

# Save to a text file
with open(metrics_file_path, "w") as file:
    file.write(metrics_text)

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test_np, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Predicted vs Actual Denormalized Values')
plt.legend()
mpld3.save_html(plt.gcf(), ".venv/residual/predicted_vs_actual.html")

train_losses = params["train_losses"]
val_losses = params["val_losses"]

# plot the training and validation loss curves
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True)  # Add grid for better readability
plt.tight_layout()
mpld3.save_html(plt.gcf(), ".venv/residual/loss_curves.html")