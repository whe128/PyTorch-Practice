import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# define input, hidden, and output dimensions
input_dim = 10    # number of features
hidden_dim = 5   # number of hidden units
output_dim = 1   # number of classes

# define the batch size
batch_size = 10

# create virtual random input data
x = torch.randn(batch_size, input_dim)
# set the target output data - shape (batch_size, output_dim)
y = torch.tensor(
    [[1.0], [0.0], [0.0], [1.0], [1.0],
     [1.0], [0.0], [0.0], [1.0], [1.0]]
)

# create sequential model, include linear, ReLu, Sigmoid
model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),       # input to hidden layer
    nn.ReLU(),                              # hidden layer activation
    nn.Linear(hidden_dim, output_dim),      # hidden to output layer
    nn.Sigmoid()                            # output layer activation
)

# define loss function and optimizer
cricriterion = nn.MSELoss()  # mean squared error loss
optimizer = optim.SGD(model.parameters(), lr=0.001)  # stochastic gradient descent

# list to store loss values
losses = []

# iterate over a number of epochs
for epoch in range(2000):
    y_pred = model(x)               # forward pass: compute predicted y
    loss = cricriterion(y_pred, y)  # compute loss
    print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')  # print loss
    optimizer.zero_grad()          # zero the gradients before backward pass
    loss.backward()                # backward pass: compute gradient
    losses.append(loss.item())     # store loss value
    optimizer.step()               # update weights

# ============================== visualization ==============================
plt.figure(figsize=(8, 5))          # create a new figure, with the size 8x5
# draw the loss curve
# x for epochs from 1 to 50
# y for loss values
plt.plot(range(1, len(losses) +1), losses, label='Training Loss') # draw the curve plot
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()                # show the legend
plt.show()
