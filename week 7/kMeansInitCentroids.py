import numpy as np

def func(X, K):

# You should return this values correctly
    centroids = np.zeros((K, X.shape[1]))


    for i in range(K):
        centroids[i,:]=np.ptp(X,axis=0)*np.random.rand(3)

    return centroids;
