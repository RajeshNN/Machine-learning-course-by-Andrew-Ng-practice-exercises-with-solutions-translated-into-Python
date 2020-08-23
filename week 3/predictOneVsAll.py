from __future__ import division
import numpy as np

def predictOVA(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]

    ## You need to return the following variables correctly
    p = np.zeros((X.shape[0], 1))

    ## Add ones to the X data matrix
    X = np.concatenate((np.ones((m, 1)), X),axis=1)
    for i in range(m):
        A=X[i,:]@(all_theta.T)
        ix=np.argmax(A)
        p[i,0]=ix+1
    return p;
