import numpy as np
# Activation function
def activate(x, W, b):
    return 1 / (1 + np.exp(- (W @ x + b)))
