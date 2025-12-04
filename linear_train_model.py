import torch                # pytorch library
import torch.nn as nn       # neural network module
import torch.optim as optim # optimization algorithms


# data
x = torch.randn(100, 1)     # 100 samples(row), 1 feature(col dimension of feature)
y = 2 * x                   # each feature is multiplied by 2 to get target

# model
model = nn.Linear(1, 1)    # linear model with 1 input and 1 output

# loss function
criterion = nn.MSELoss()     # MSE: mean squared error

# input the linear model parameters to the optimizer
optimizer = optim.SGD(model.parameters(), lr = 0.1) # SGD: stochastic gradient descent

# train loop
for epoch in range(200):
    y_pred = model(x)            # use the linear model(1 in and 1 out) to predict y from x

    # loss tensor includes w, b , sample yi, xi
    loss = criterion(y_pred, y)  # loss between predicted and acutal target
    optimizer.zero_grad()        # clean gradients
    loss.backward()              # backpropagation, loss is also a function of model parameters
    optimizer.step()             # update model parameters

# test
print("Predict 2 →", model(torch.tensor([[2.0]])))
