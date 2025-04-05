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

directory = r"C:\Users\alexa\PycharmProjects\CapstoneProject\.venv"
data_dir = os.path.join(directory, 'data', 'Constellation (65,536)')
results_dir = os.path.join(directory, 'model results', 'Constellation')

all_files = os.listdir(data_dir)
dataset_names = sorted(set(f.split('_')[0] for f in all_files if f.endswith("_input_complex.csv")))

#instantiate scaling factor
scaling_factor = 1.0

def load_data(filepath):
    return pd.read_csv(filepath, sep=',', header=None).to_numpy()

def plot_two_lines(x1, y1, x2, y2, title, x_label, y_label, label1, label2, save_path):
    """
    Creates an interactive Plotly plot for two solid lines over the same x-axis and saves it as an HTML file.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name=label1, line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=x2, y=y2, mode='lines', name=label2, line=dict(color='red', width=2)))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white", hovermode="x")
    fig.write_html(save_path)

# Dictionary to store processed datasets, including scaling factor
processed_datasets = {}

# Iterate over each dataset
for dataset in dataset_names:
    print(f"\n🔄 Processing dataset: {dataset}...")

    input_file = os.path.join(data_dir, f"{dataset}_input_complex.csv")
    output_file = os.path.join(data_dir, f"{dataset}_output_complex.csv")

    # Load dataset
    input_data = load_data(input_file)
    output_data = load_data(output_file)

    # Truncate if input and output are different lengths
    if len(input_data) != len(output_data):
        min_length = min(len(input_data), len(output_data))
        input_data = input_data[:min_length]
        output_data = output_data[:min_length]

    # NaN value check on input data
    for i in range(1, 3):  # Columns 1 and 2 (I and Q)
        if np.isnan(input_data[:, i]).any():
            input_mean = np.nanmean(input_data[:, i])
            input_data[np.isnan(input_data[:, i]), i] = input_mean

    #NaN value check on output data
    for i in range(1, 3):  # Columns 1 and 2 (I and Q)
        if np.isnan(output_data[:, i]).any():
            output_mean = np.nanmean(output_data[:, i])
            output_data[np.isnan(output_data[:, i]), i] = output_mean

    # Compute scaling factor
    scaling_factor = int(np.max(input_data[:, 1]) / np.max(output_data[:, 1]))

    # Combine data
    data = np.column_stack((
        input_data[:, 0],   # Time
        input_data[:, 1],   # Input I
        input_data[:, 2],   # Input Q
        output_data[:, 1] * scaling_factor,    # Output I (scaled)
        output_data[:, 2] * scaling_factor     # Output Q (scaled)
    ))

    # Store processed data and scaling factor in a dictionary
    processed_datasets[dataset] = {
        "data": data,               # Store full dataset (all 5 columns)
        "scaling_factor": scaling_factor  # Store computed scaling factor
    }

    print(f"✅ {dataset} loaded and stored successfully!")

# Define window and step sizes
window_size = 512
step_size = 1  # Adjust step size to control overlap

class ModelDataset(Dataset):
    def __init__(self, data, window_size, step_size):
        """
        Args:
        - data (numpy array): 2D array of shape (N, 5) with columns [Time, I_in, Q_in, I_out, Q_out]
        - window_size (int): Number of time steps in each window
        - step_size (int): Step size between windows
        """
        self.data = data
        self.window_size = window_size
        self.step_size = step_size
        self.num_windows = (len(self.data) - self.window_size) // self.step_size + 1

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        """
        Returns a single window of data.

        - X (torch.Tensor): Full input sequence (window_size,)
        - y (torch.Tensor): Target (middle value of output signal)
        - t (float): Time index of the middle point
        """
        start_idx = idx * self.step_size
        end_idx = start_idx + self.window_size
        window = self.data[start_idx:end_idx]

        # Extract input, output, and time signals
        time_signal = window[:, 0]  # Extract time column
        input_I = window[:, 1]
        input_Q = window[:, 2]
        output_I = window[:, 3]
        output_Q = window[:, 4]

        # Find middle index
        middle_index = len(output_I) // 2
        middle_time = time_signal[middle_index]  # Time at middle of the window

        # Convert to PyTorch tensors
        input_tensor = torch.tensor(np.stack([input_I, input_Q], axis = 1), dtype=torch.float32)  # Full input sequence
        target_tensor = torch.tensor([output_I[middle_index], output_Q[middle_index]], dtype=torch.float32)  # Middle output

        return input_tensor, target_tensor, middle_time

class LogCoshLoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.mean(torch.log(torch.cosh(y_pred - y_true + 1e-12)))  # Avoid log(0)


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

# Define the Residual Connections model
class FiberOpticFNN5(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FiberOpticFNN5, self).__init__()
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

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, patience):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_losses = []
    val_losses = []
    best_weights = None

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for X_batch, y_batch, _ in train_loader:

            # Move data to GPU
            X_batch = X_batch.to(device).view(X_batch.shape[0], -1)
            y_batch = y_batch.to(device)

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
            for X_batch, y_batch, _ in val_loader:

                # Move data to GPU
                X_batch = X_batch.to(device).view(X_batch.shape[0], -1)
                y_batch = y_batch.to(device)

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

     # Evaluate on the test set
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch, _ in test_loader:
            X_batch = X_batch.to(device).view(X_batch.shape[0], -1)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            y_true.append(y_batch.cpu().numpy())
            y_pred.append(predictions.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    return train_losses, val_losses, best_weights, y_true, y_pred

def objective(trial, dataset_name, model_name, model_class):
    num_epochs = 100
    best_val_loss = float("inf")

    # Define dataset-specific directory
    model_dir = os.path.join(results_dir, dataset_name, model_name)
    os.makedirs(model_dir, exist_ok=True)

    data = processed_datasets[dataset_name]["data"]
    scaling_factor = processed_datasets[dataset_name]["scaling_factor"]

    # Create dataset instance
    dataset = ModelDataset(data, window_size, step_size)

    # Split dataset into train, validation, and test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Define DataLoaders with shuffle only for training
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Hyperparameters to tune
    hidden_dim = trial.suggest_int("hidden dim", 256, 640, step=32)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight decay", 1e-6, 1e-3, log=True)

    # Model arguments dictionary
    model_args = {
        "input_dim": window_size * 2,
        "hidden_dim": hidden_dim,
        "output_dim": 2
    }

    if "dropout" in model_class.__init__.__code__.co_varnames:
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        model_args["dropout"] = dropout_rate

    model = model_class(**model_args)
    criterion = nn.MSELoss()  # Define loss function
    #criterion = nn.SmoothL1Loss(beta=1e-4)  # Huber Loss with a small beta
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Define optimizer

    # Train the model
    train_losses, val_losses, best_weights, y_true, y_pred = train_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        patience=10
    )

    if min(val_losses) < best_val_loss:
        best_val_loss = min(val_losses)

        #Save model weights
        weights_path = os.path.join(model_dir, "model_weights.pth")
        torch.save(best_weights, weights_path)

        # Save best params
        best_params = {
            "hidden_dim": hidden_dim,
            "lr": lr,
            "weight_decay": weight_decay,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "scaling_factor": scaling_factor
        }
        if "dropout" in model_args:
            best_params["dropout"] = dropout_rate

        params_path = os.path.join(model_dir, "best_params.json")
        with open(params_path, "w") as json_file:
            json.dump(best_params, json_file, indent=4)
        print(f"Files saved successfully: {weights_path}, {params_path}")


        # Save loss curves as an interactive HTML file
        plot_two_lines(x1 = np.arange(len(train_losses)), y1 = train_losses,
                       x2 = np.arange(len(val_losses)), y2 = val_losses,
                       title = "Training vs Validation Loss",
                       x_label="Epoch", y_label="Loss",
                       label1= "Train Loss", label2 = "Val Loss",
                       save_path= os.path.join(model_dir, "loss_plot.html")
                       )

        # Save Prediction vs Actual Output Plot as interactive HTML files
        plot_two_lines(x1 = data[:,0], y1 = y_true[:, 0],
                       x2 = data[:,0], y2 = y_pred[:, 0],
                       title = "Predictions vs Actual (I Component)",
                       x_label="Time", y_label="Amplitude",
                       label1 = "Actual (I Component)", label2 = "Predicted (I Component)",
                       save_path= os.path.join(model_dir, "predicted_vs_actual_I.html")
                       )

        plot_two_lines(x1=data[:, 0], y1=y_true[:, 1],
                       x2=data[:, 0], y2=y_pred[:, 1],
                       title="Predictions vs Actual (Q Component)",
                       x_label="Time", y_label="Amplitude",
                       label1="Actual (Q Component)", label2="Predicted (Q Component)",
                       save_path=os.path.join(model_dir, "predicted_vs_actual_Q.html")
                       )

        # Compute & Save Metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            "MSE": mse,
            "MAE": mae,
            "R2 Score": r2
        }

        metrics_path = os.path.join(model_dir, "metrics.json")
        with open(metrics_path, "w") as json_file:
            json.dump(metrics, json_file, indent=4)


    return best_val_loss


# Dictionary mapping model names to their respective classes
model_info = {
    "Basic": FiberOpticFNN0,
    "Deeper": FiberOpticFNN1,
    "Wider": FiberOpticFNN2,
    "Dynamic": FiberOpticFNN3,
    "Noise Resilient": FiberOpticFNN4,
    "Residual": FiberOpticFNN5
}
for dataset_name in processed_datasets.keys():
    print(f"\n🔄 Processing dataset: {dataset_name}...")
    # Run Optuna study for each model
    for model_name, model_class in model_info.items():
        print(f"🔄 Running Optuna for {model_name} model...")

        # Create an Optuna study
        study = optuna.create_study(direction="minimize")

        # Optimize using lambda to pass additional arguments
        study.optimize(lambda trial: objective(trial, dataset_name, model_name, model_class), n_trials=18)

        print(f"✅ Finished optimization for {model_name}\n")
