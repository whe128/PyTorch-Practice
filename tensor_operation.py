import torch

#set data type and device

# tensort data type is float32
dtype = torch.float

# use the cpu for computation
device = torch.device("cpu")

# create two random tensors
# shape of tensor is 2x3 (2 samples, 3 features)
a = torch.randn(2, 3, dtype=dtype, device=device)
b = torch.randn(2, 3, dtype=dtype, device=device)

print("Tensor a:\n", a)
print("Tensor b:\n", b)

# tensor multiplication (element-wise)
c = a * b
print("a * b (element-wise multiplication):\n", c)

# tensor addition
d = a + b
print("a + b (addition):\n", d)

# output tensor sample 2, feature 3
print("Sample 2, Feature 3 of tensor a:", a[1, 2])

# output max value of tensor a
print("Max value of tensor a:", torch.max(a))


print(torch.rand(5, 3))
