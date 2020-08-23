import numpy as np

def func(X, p):

# You need to return the following variables correctly.
    x=np.ravel(X)
    X_poly = np.zeros((len(x), p))

 
    for i in range(len(x)):
        for j in range(p):
            X_poly[i,j]=X[i]**(j+1)
    return X_poly;
