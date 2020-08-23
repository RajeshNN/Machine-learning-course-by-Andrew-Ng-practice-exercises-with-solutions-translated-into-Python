import matplotlib as m
import matplotlib.pyplot as plt
import numpy as np

def func(X, idx, K, ax=None, fig=None):
#PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
#index assignments in idx have the same color
#   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those 
#   with the same index assignments in idx have the same color

# Create palette
    if not fig:
        fig,ax=plt.subplots(1)
    palette = np.random.rand(K,3)

    for i in range(idx.shape[0]):
        for j in range(K):
            if idx[i]==j:
                ax.scatter(X[i,0], X[i,1], edgecolor='k', color=palette[j,:])

    return fig,ax;
    
