from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import lrCostFunc
import lrgradFunc
import scipy.optimize as sc

def oneVsAll(X, y, num_labels, lambdaa):
    # Some useful variables
    [m,n] = X.shape

    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.concatenate((np.ones((m, 1)), X),axis=1)

    initial_theta=np.zeros((n+1,1))

    for c in range(1,num_labels+1):
        yk=np.zeros((y.shape[0],1))
        print('c=',c)
        for i in range(m):
            if y[i,0]==c:
                yk[i,0]=1
    
        res= sc.minimize(lrCostFunc.lrCostFunction, initial_theta, args=(X,yk,lambdaa),\
                        method='CG',jac=lrgradFunc.lrgradFunc)
        print(res.message)
        all_theta[c-1,:]=res.x.reshape(1,-1)
        
        
    return all_theta;
