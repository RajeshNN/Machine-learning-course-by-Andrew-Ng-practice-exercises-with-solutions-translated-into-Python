import numpy as np
import matplotlib.pyplot as plt
import plotData
import gaussianKernelGramMatrix

def func(X, y, model):
#VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
#SVM
#   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary 
#   learned by the SVM and overlays the data on it
    
    fig,ax,a,b=plotData.func(X,y)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    
    if model.kernel=='precomputed':
        
        z=gaussianKernelGramMatrix.func(xy,X)

        Z = model.decision_function(z).reshape(XX.shape)
    else:

        Z=model.decision_function(xy).reshape(XX.shape)

    Z=Z.T                                          
    ax.contour(xx, yy, Z, colors='k', levels=[-0.5, 0, 0.5],\
               linestyles=['--','-','--'])
    return fig,ax,a,b;
