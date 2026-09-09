import numpy as np

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

        # cross entropy loss
        neg_log_loss = -np.log(correct_confidence)
        return neg_log_loss

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # If labels are one-hot encoded
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()



class Accuracy(Loss):
    def forward(self, ypred, ytrue):
        predictions = np.argmax(ypred, axis=1)

        # if target are one-hot encoded
        if len(ytrue.shape) == 2:
            ytrue = np.argmax(ytrue, axis=1)

        sample_losses = predictions == ytrue

        return sample_losses
