import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Load data
input_data = pd.read_csv(".venv/OOK_input_data.txt", delim_whitespace=True, header=None, names=["Time", "Amplitude"])
output_data = pd.read_csv(".venv/OOK_output_data.txt", delim_whitespace=True, header=None, names=["Time", "Amplitude"])

# Normalize data
scaler_input = MinMaxScaler()
scaler_output = MinMaxScaler()
input_data_normalized = scaler_input.fit_transform(input_data[["Amplitude"]])
output_data_normalized = scaler_output.fit_transform(output_data[["Amplitude"]])

# Split data
data_length = len(input_data_normalized)
train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
train_end = int(train_ratio * data_length)
val_end = int((train_ratio + val_ratio) * data_length)

X_train = input_data_normalized[:train_end]
y_train = output_data_normalized[:train_end]
X_val = input_data_normalized[train_end:val_end]
y_val = output_data_normalized[train_end:val_end]
X_test = input_data_normalized[val_end:]
y_test = output_data_normalized[val_end:]

# Create sequences
def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

seq_length = 100
X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

# Datasets and DataLoaders
train_dataset = TensorDataset(X_train_seq, y_train_seq)
val_dataset = TensorDataset(X_val_seq, y_val_seq)
test_dataset = TensorDataset(X_test_seq, y_test_seq)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Model definition
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_activation = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.hidden_activation(lstm_out)  # Apply ReLU to LSTM output
        last_inner_state = lstm_out[:, -1, :]  # Take the last output
        out = self.fc(last_inner_state)
        return out

# Weight initialization
def init_weights(model):
    for name, param in model.named_parameters():
        if "weight" in name and "lstm" in name:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)

model = LSTMModel(1, 256, 5, 1)
init_weights(model)

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss = evaluate_model(model, val_loader, criterion)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

def evaluate_model(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            val_loss += loss.item()
    return val_loss / len(loader)

# Train and evaluate
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)
test_loss = evaluate_model(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.6f}")

# Plot predictions
def plot_predictions(model, loader, scaler_output):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch)
            predictions.append(preds.numpy())
            actuals.append(y_batch.numpy())
    predictions = np.concatenate(predictions).reshape(-1, 1)
    actuals = np.concatenate(actuals).reshape(-1, 1)
    predictions = scaler_output.inverse_transform(predictions)
    actuals = scaler_output.inverse_transform(actuals)
    plt.plot(actuals, label="True Values", alpha=0.8)
    plt.plot(predictions, label="Predictions", alpha=0.6)
    plt.legend()
    plt.show()

plot_predictions(model, test_loader, scaler_output)
