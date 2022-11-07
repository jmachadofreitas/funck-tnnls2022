import numpy as np


def accuracy(y, y_logits):
    y_ = (y_logits > 0.0).astype(np.float32)
    return np.mean((y_ == y).astype(np.float32))
