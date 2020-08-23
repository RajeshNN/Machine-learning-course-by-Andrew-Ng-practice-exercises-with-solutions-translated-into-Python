from __future__ import division
import numpy as np
import sigmoid

def predict(theta,X):
    m = X.shape[0]
    p = np.zeros((m, 1))
    q=X@theta
    a=sigmoid.sigmoid(q)
    for i in range(m):
        if a[i,0]>=0.5:
            p[i,0]=1
        else:
            p[i,0]=0
    return p;
