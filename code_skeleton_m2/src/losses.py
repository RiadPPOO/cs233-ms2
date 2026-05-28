import numpy as np

class MSE:
    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        return np.mean((y_pred - y_true) ** 2)


    @staticmethod
    def gradient(y_true, y_pred):
        N = y_true.shape[0]
        return 2 * (y_pred - y_true) / N
    

class CrossEntropy:
    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Predicted probabilities (after softmax)
        :return: (flt)
        """
        # Clip for numerical stability (avoid log(0))
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        N = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred)) / N
 
    @staticmethod
    def gradient(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        N = y_true.shape[0]
        return -(y_true / y_pred) / N
 
