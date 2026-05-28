import numpy as np

class Sigmoid:
    @staticmethod
    def forward(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def gradient(z):
        s = Sigmoid.forward(z)
        return s * (1 - s)

class ReLU:
    @staticmethod
    def forward(z):
        return np.maximum(0, z)

    @staticmethod
    def gradient(z):
        return (z > 0).astype(float)

class Softmax:
    @staticmethod
    def forward(z):
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    @staticmethod
    def gradient(z):
        s = Softmax.forward(z)
        return s * (1 - s)

class Identity:
    @staticmethod
    def forward(z):
        return z

    @staticmethod
    def gradient(z):
        return np.ones_like(z)