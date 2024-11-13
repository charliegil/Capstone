import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F  # for one-hot encoding
from torch.utils.data import DataLoader, Dataset
import torchvision
import numpy as np
import matplotlib.pyplot as plt


class CTDataset(Dataset):
    def __init__(self, filepath):
        self.x, self.y = torch.load(filepath)
        self.x = self.x / 255.0  # normalize pixel intensity from [0, 255] to [0, 1]
        self.y = F.one_hot(self.y, num_classes=10).to(float)  # one-hot encoding of labels

    def __len__(self):
        return self.x.shape[0]  # number of images

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(28 ** 2, 100)  # input layer -> 100 output neurons
        self.Matrix2 = nn.Linear(100, 50)  # first hidden layer
        self.Matrix3 = nn.Linear(50, 10)  # second hidden layer/output layer
        self.R = nn.ReLU()  # activation  function

    # Forward data through network
    def forward(self, x):
        x = x.view(-1, 28**2)  # -1 -> keep same batch size, 28**2 -> convert 28 * 28 image to vector
        x = self.R(self.Matrix1(x))  # pass x through first layer and apply activation function
        x = self.R(self.Matrix2(x))  # pass x through second layer and apply activation function
        x = self.Matrix3(x)  # pass x through last layer
        return x.squeeze()  # squeeze tensor down to a single dimension


def train_model(dl, f, num_epochs):
    # Define backpropagation and loss function
    opt = SGD(f.parameters(), lr=0.01)  # define backpropagation function
    L = nn.CrossEntropyLoss()  # define Cross Entropy Loss function

    # Train model
    losses = []  # keep track of
    epochs = []  # store data for each epoch

    for epoch in range(num_epochs):  # pass data through network once for each epoch
        print(epoch)
        N = len(dl)  # total number of batches

        for i, (x, y) in enumerate(dl):  # loop through each batch
            # Update network weights
            opt.zero_grad()  # reset gradients
            loss_value = L(f(x), y)  # compute loss value
            loss_value.backward()  # back-propagate loss value
            opt.step()  # adjust weights

            # Store training data
            epochs.append(epoch+i/N)
            losses.append(loss_value.item())

    return np.array(epochs), np.array(losses)


if __name__ == '__main__':
    # x -> features, y -> labels
    # x, y = torch.load('C:\\Users\\Charlie\\Documents\\MNIST\\processed\\training.pt')

    # Convert tensor to numpy array (for future reference)
    # x[2].numpy()

    # One-hot encoding of labels for classification
    # y_new = F.one_hot(y, num_classes=10)

    # Convert matrix to vector (28 x 28 image becomes a vector)
    # x.view(-1, 28**2)

    # Get training and testing data
    train_dataset = CTDataset('C:\\Users\\Charlie\\Documents\\MNIST\\processed\\training.pt')
    test_dataset = CTDataset('C:\\Users\\Charlie\\Documents\\MNIST\\processed\\test.pt')

    # Define batch size
    batch_size = 5  # experiment with size?

    # Create dataloader for training
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    # Define network
    f = NN()

    # Train network
    epoch_data, loss_data = train_model(train_loader, f, 1)

    # Average results per epoch
    epoch_data_avg = epoch_data.reshape(20, -1).mean(axis=1)
    loss_data_avg = loss_data.reshape(20, -1).mean(axis=1)

    # Plot loss function for each batch
    plt.plot(epoch_data_avg, loss_data_avg)
    plt.xlabel('Epoch Number')
    plt.ylabel('Cross Entropy')
    plt.title('Cross Entropy (average per epoch)')
    plt.show()

    # Test trained model on training data
    xs, ys = train_dataset[0:2000]

    yhats_train = f(xs).argmax(axis=1)  # compute predicted labels of first 2000 values of training data

    fig, ax = plt.subplots(10, 4, figsize=(10, 15))
    for i in range(40):
        plt.subplot(10, 4, i+1)
        plt.imshow(xs[i])
        plt.title(f'Predicted Digit: {yhats_train[i]}')
    fig.tight_layout()
    plt.show()

    # Test trained model on testing data

    xs, ys = test_dataset[0:2000]

    yhats_test = f(xs).argmax(axis=1)  # compute predicted labels of first 2000 values of training data

    # Plot 40 predictions
    fig, ax = plt.subplots(10, 4, figsize=(10, 15))
    for i in range(40):
        plt.subplot(10, 4, i + 1)
        plt.imshow(xs[i])
        plt.title(f'Predicted Digit: {yhats_test[i]}')
    fig.tight_layout()
    plt.show()


