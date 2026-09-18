import numpy as np


class Adam:
    def __init__(self, parameters, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = [np.zeros_like(param) for param, _ in parameters]
        self.v = [np.zeros_like(param) for param, _ in parameters]

    def step(self):
        self.t += 1
        for i, (param, grad) in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def zero_grad(self):
        for _, grad in self.parameters:
            grad.fill(0.0)


def clip_grad_norm(parameters, max_norm=1.0):
    total_sq = 0.0
    for _, grad in parameters:
        total_sq += np.sum(grad ** 2)

    total_norm = float(np.sqrt(total_sq))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        for _, grad in parameters:
            grad *= scale

    return total_norm
