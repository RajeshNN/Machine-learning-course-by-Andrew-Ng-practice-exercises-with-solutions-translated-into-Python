import numpy as np

def func(X, centroids):

# Set K
    K = centroids.shape[0]

# You need to return the following variables correctly.
    idx = np.zeros((X.shape[0], 1))

    for i in range(X.shape[0]):
        q=np.zeros((centroids.shape[0],1))
        for p in range(centroids.shape[0])  :
            for j in range(centroids.shape[1]):
                q[p]+=(X[i,j]-centroids[p,j])**2
        idx[i]=np.argmin(q)

    return idx;
