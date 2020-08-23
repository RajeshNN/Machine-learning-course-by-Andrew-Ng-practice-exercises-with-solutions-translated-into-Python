import numpy as np
import gaussianKernel

def func(X1,X2, n=0.1):
    """(Pre)calculates Gram Matrix K"""

    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            gram_matrix[i, j] = gaussianKernel.func(x1, x2, n)
    return gram_matrix;
