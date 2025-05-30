import numpy as np
from yara.nn import DenseLayer, ActivationReLU, ActivationSoftmax, Loss_CategoricalCrossEntropy
from nnfs.datasets import spiral_data


x, y = spiral_data(samples=100, classes=3)

dense1 = DenseLayer(2, 3)
activation1 = ActivationReLU()
dense2 = DenseLayer(3, 3)
activation2 = ActivationSoftmax()
loss_function = Loss_CategoricalCrossEntropy()

dense1.forward(x)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = loss_function.calculate(activation2.output, y)

print(activation2.output[:5])
print(f"Loss = {loss}")
