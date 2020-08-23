import matplotlib.pyplot as plt
import plotDataPoints
import drawLine

def func(X, centroids, previous, idx, K, i, fig=None, ax=None):

# Plot the examples

    fig,ax=plotDataPoints.func(X, idx, K, ax, fig)

# Plot the centroids as black x's
    ax.plot(centroids[:,0], centroids[:,1],'x',markeredgecolor='k', markersize=10, linewidth=3)

# Plot the history of the centroids with lines
    for j in range(centroids.shape[0]):
        drawLine.func(centroids[j, :], previous[j, :],ax)

# Title
    plt.title("Iteration number %i"%(i+1))

    plt.show()
    return fig, ax;
