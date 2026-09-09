from yara.nn import Value

x1 = Value(2.0)
x2 = Value(3.0)

a = x1*x2
b = a + Value(1.0)

y = b.relu()

y.backward()

print(f"y_data = {y.data}")
print(f"y = {y}")

print(f"x1.grad = {x1.grad}")
print(f"x2.grad = {x2.grad}")

# TASK 1
x = Value(2.0)
f = x ** 3
f.backward()

print(f"x.grad = {x.grad}")

# USING PYTORCH
# import torch

# x1_torch = torch.tensor([2.0], requires_grad=True)
# x2_torch = torch.tensor([3.0], requires_grad=True)

# a_torch = x1_torch * x2_torch
# b_torch = a_torch + torch.tensor([1.0])
# y_torch = b_torch.relu()

# y_torch.backward()

# print(f"y_torch_data = {y_torch.data}")
# print(f"y_torch = {y_torch}")
# print(f"x1_torch.grad = {x1_torch.grad}")
# print(f"x2_torch.grad = {x2_torch.grad}")
