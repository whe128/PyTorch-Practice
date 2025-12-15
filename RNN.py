import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

#============= RNN model definition
class  SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # input_size: number of features per time step
        # hidden_size: dimension of the RNN hidden state
        # output_size: dimension of the final output
        super(SimpleRNN, self).__init__()
        # RNN layer
        # ht = tanh(Wxhxt + Whhht-1 + bh)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        # Fully connected layer
        # yt = Why ht + by
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x   shape: (batch_size, seq_length, input_size)
        # each sample has seq_length time steps,
        # each time step has input_size features

        # x.shape = (batch_size, seq_length, input_size)
        # out.shape = (batch_size, seq_length, hidden_size)
        out, _ = self.rnn(x)            # _: hidden state, don't need it here

        # use the last time step's output for output
        # out.shape = (batch_size, hidden_size), dimension 1 is removed
        out = out[:, -1, :]

        # full connection
        out = self.fc(out)
        return out

# ============= generate training data
num_samples = 1000
seq_len = 10
input_size = 5
output_size = 2         # binary classification

# random input data (batch_size, seq_length, input_size)
X = torch.randn(num_samples, seq_len, input_size)
# random labels (0 or 1)   (batch_size, output_size)   range[0, output_size-1]
Y = torch.randint(0, output_size, (num_samples,))    # single tuple, need comma at the end

# create dataset and dataloader
dataset = TensorDataset(X, Y)
train_loader = DataLoader(dataset, batch_size = 32, shuffle = True)

# ============= create model, define loss function and optimizer
model = SimpleRNN(input_size = input_size, hidden_size = 64, output_size = output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============= train the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumulate loss
        total_loss += loss.item()

        # calculate accuracy
        _, predicted = torch.max(outputs, 1)   # 1 for each row
        total += labels.size(0)                # batch size
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# ============= evaluate the model
model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    for inputs, labels in train_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
