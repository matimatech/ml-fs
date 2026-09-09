import numpy as np
from yara.nn import DenseLayer, ActivationReLU, ActivationSoftmax, Loss_CategoricalCrossEntropy, Accuracy
# from nnfs.datasets import spiral_data
# import torch

# x, y = spiral_data(samples=100, classes=3)
x = np.random.rand(3, 2)
y = np.random.rand(3)
print(x, y)

dense1 = DenseLayer(2, 3)
activation1 = ActivationReLU()
dense2 = DenseLayer(3, 3)
activation2 = ActivationSoftmax()
loss_function = Loss_CategoricalCrossEntropy()
accuracy = Accuracy()

dense1.forward(x)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = loss_function.calculate(activation2.output, y)
acc = accuracy.calculate(activation2.output, y)

print(activation2.output[:5])
print(f"Loss = {loss}")
print(f"Acc = {acc}")


softmax_outputs = np.array([[0.7, 0.2, 0.1],
                            [0.5, 0.1, 0.4],
                            [0.02, 0.01, 0.97]])
# Target (ground-truth) labels for 3 samples
class_targets = np.array([0, 1, 1])


predictions = np.argmax(softmax_outputs, axis=1)
print(predictions)