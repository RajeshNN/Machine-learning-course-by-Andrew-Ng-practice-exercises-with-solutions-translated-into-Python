import numpy as np

def func(Z, U, K):

    X_rec = np.zeros((Z.shape[0],U.shape[0]))
               
    U=U[:,range(K)]
    X_rec=Z@U.T

    return X_rec;
