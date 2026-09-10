class GradientDescent:
    """
    Gradient descent optimizer implementation.
    Parameters
    ----------
    learning_rate : float, optional
        Learning rate for the optimizer, by default 0.01.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, params, grads):
        return [p - self.learning_rate * g for p, g in zip(params, grads)]

class SGDMomentum:
    """
    SGD with momentum optimizer implementation.
    Parameters
    ----------
    learning_rate : float, optional
        Learning rate for the optimizer, by default 0.01.
    momentum : float, optional
        Momentum for the optimizer, by default 0.9.
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [0.0] * len(params)

        self.velocity = [self.momentum * v + self.learning_rate * g for v, g in zip(self.velocity, grads)]
        return [p - v for p, v in zip(params, self.velocity)]

class Adam:
    """
    Adam optimizer implementation.
    Parameters
    ----------
    learning_rate : float, optional
        Learning rate for the optimizer, by default 0.01.
    beta1 : float, optional
        Beta1 for the optimizer, by default 0.9.
    beta2 : float, optional
        Beta2 for the optimizer, by default 0.999.
    epsilon : float, optional
        Epsilon for the optimizer, by default 1e-8.
    """
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1
        self.m = [self.beta1 * m + (1 - self.beta1) * g for m, g in zip(self.m, grads)]
        self.v = [self.beta2 * v + (1 - self.beta2) * (g ** 2) for v, g in zip(self.v, grads)]
        m_hat = [m / (1 - self.beta1 ** self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2 ** self.t) for v in self.v]

        return [p - self.learning_rate * m_hat / (v_hat ** 0.5 + self.epsilon) for p, m_hat, v_hat in zip(params, m_hat, v_hat)]
