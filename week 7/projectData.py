import numpy as np

def func(X, U, K):

    Z = np.zeros((X.shape[0], K))

    U=U[:,range(K)]
    Z=X@U

    return Z;
