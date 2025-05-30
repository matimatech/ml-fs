import numpy as np


class DenseLayer:
    """
    Fully connected layers
    """

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias


class ActivationReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        prob_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = prob_values

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, ypred, ytrue):
        samples = len(ypred)

        ypred_clipped = np.clip(ypred, 1e-7, 1 - 1e-7)

        if len(ytrue.shape) == 1:
            correct_confidence = ypred_clipped[
                range(samples),
                ytrue
            ]
        # for one hot encoded label
        elif len(ytrue.shape) == 2:
            correct_confidence = np.sum(
                ypred_clipped * ytrue,
                axis=1
            )

        neg_log_loss = -np.log(correct_confidence)
        return neg_log_loss






