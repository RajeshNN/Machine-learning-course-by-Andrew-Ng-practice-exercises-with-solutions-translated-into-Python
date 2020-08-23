import numpy as np
import statistics as st
def func(X):
    X_norm=np.zeros((X.shape))
    mu=np.zeros((1,X.shape[1]))
    sigma=np.zeros((1,X.shape[1]))
    for i in range(X.shape[1]):
        mu[0,i]=np.sum(X[:,i])/X.shape[0]
        sigma[0,i]=st.stdev(X[:,i])
        X_norm[:,i]=(X[:,i]-mu[0,i])/sigma[0,i]
    return X_norm, mu, sigma;
