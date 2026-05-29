import numpy as np
from ..activations import Sigmoid, ReLU, Softmax, Identity


class MLP:
    def __init__(self, dimensions, activations, class_weights=None, weight_decay=0.0):
        """
        :param dimensions: list (input, hidden, ..., hidden, output)
        :param activations: list of N-1 activation classes
        :param class_weights: optional (n_classes,) per-class weights
        :param weight_decay: L2 regularization coefficient lambda (default 0 = off)
        """
        self.dimensions = list(dimensions)
        self.activations = list(activations)
        self.n_layers = len(self.dimensions) - 1
        self.class_weights = class_weights
        self.weight_decay = weight_decay

        self.w = {}
        self.b = {}
        for l in range(1, self.n_layers + 1):
            fan_in = self.dimensions[l - 1]
            fan_out = self.dimensions[l]
            if self.activations[l - 1] is ReLU:
                std = np.sqrt(2.0 / fan_in)
            else:
                std = np.sqrt(1.0 / fan_in)
            self.w[l] = np.random.randn(fan_in, fan_out) * std
            self.b[l] = np.zeros((1, fan_out))

        self.lr = 1e-3

    def feed_forward(self, x):
        z = {}
        a = {0: x}
        for l in range(1, self.n_layers + 1):
            z[l] = a[l - 1] @ self.w[l] + self.b[l]
            a[l] = self.activations[l - 1].forward(z[l])
        return z, a

    def predict(self, x):
        _, a = self.feed_forward(x)
        return a[self.n_layers]

    def back_prop(self, z, a, y_true, loss):
        L = self.n_layers
        N = y_true.shape[0]
        out_act = self.activations[L - 1]

        if out_act is Softmax:
            if self.class_weights is not None:
                sample_w = (y_true @ self.class_weights).reshape(-1, 1)
                delta = sample_w * (a[L] - y_true) / N
            else:
                delta = (a[L] - y_true) / N
        else:
            delta = loss.gradient(y_true, a[L]) * out_act.gradient(z[L])

        for l in range(L, 0, -1):
            dw = a[l - 1].T @ delta
            if l > 1:
                delta_prev = (delta @ self.w[l].T) * self.activations[l - 2].gradient(z[l - 1])
            self.update_w_b(l, dw, delta)
            if l > 1:
                delta = delta_prev

    def update_w_b(self, index, dw, delta):
        # L2 weight decay: add lambda * W to the weight gradient (NOT to bias)
        if self.weight_decay > 0:
            dw = dw + self.weight_decay * self.w[index]
        self.w[index] = self.w[index] - self.lr * dw
        self.b[index] = self.b[index] - self.lr * np.sum(delta, axis=0, keepdims=True)

    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate=1e-3):
        self.lr = learning_rate
        N = x.shape[0]
        if batch_size is None or batch_size > N:
            batch_size = N

        for epoch in range(epochs):
            perm = np.random.permutation(N)
            x_sh = x[perm]
            y_sh = y_true[perm]
            for start in range(0, N, batch_size):
                xb = x_sh[start:start + batch_size]
                yb = y_sh[start:start + batch_size]
                z, a = self.feed_forward(xb)
                self.back_prop(z, a, yb, loss)