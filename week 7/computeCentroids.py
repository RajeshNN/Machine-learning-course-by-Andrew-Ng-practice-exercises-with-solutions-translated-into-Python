import numpy as np

def func(X, idx, K):

# Useful variables
    m, n = X.shape

# You need to return the following variables correctly.
    centroids = np.zeros((K, n))

    P=np.zeros((K,1))
    for i in range(K):
        for j in range(m):
            if idx[j]==i:
                for l in range(n):
                    centroids[i,l]+=X[j,l]

                P[i]=P[i]+1

    for m in range(K):
        if P[m]:
            centroids[m,:]=(centroids[m,:])/P[m]

    return centroids;
