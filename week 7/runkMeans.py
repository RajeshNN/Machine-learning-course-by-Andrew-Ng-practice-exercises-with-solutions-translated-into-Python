import numpy as np
import matplotlib.pyplot as plt
import findClosestCentroids
import plotProgresskMeans
import computeCentroids

def func(X, initial_centroids, max_iters, plot_progress='False'):

# Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))
    fig=None
# Run K-Means
    for i in range(max_iters):

        print('K-Means iteration',i+1,'/',max_iters,'...\n')
    
    # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids.func(X, centroids)
            
    # Optionally, plot progress here
        if plot_progress=='True':
            if fig:
                dummy=plt.figure()
                m=dummy.canvas.manager
                m.canvas.figure=fig
                fig.set_canvas(m.canvas)
                fig,ax=plotProgresskMeans.func(X, centroids, previous_centroids, idx, K, i, fig=fig,ax=ax)
                previous_centroids = centroids
            else:
                fig,ax=plotProgresskMeans.func(X, centroids, previous_centroids, idx, K, i)
                previous_centroids = centroids

    # Given the memberships, compute new centroids
        centroids = computeCentroids.func(X, idx, K)

    return centroids, idx;
