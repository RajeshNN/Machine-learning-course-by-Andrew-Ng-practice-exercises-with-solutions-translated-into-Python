import numpy as np

def func(X):

    m, n = X.shape

    C= np.zeros((n,n))
    for i in range(m):
        C+=np.array(np.matrix(X[i,:])).T@np.array(np.matrix(X[i,:]))
    C=C/m
    U,S,V=np.linalg.svd(C)
    return U, S;
