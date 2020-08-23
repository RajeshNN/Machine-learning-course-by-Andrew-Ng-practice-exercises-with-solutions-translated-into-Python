from __future__ import division
import numpy as np
import scipy.optimize as sc
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import plotData
import mapFeature

def plotDecisionBoundary(theta, X, y):
    # Plot Data
    f,j1,a,b=plotData.plotData(X[:,1:3], y)

    if X.shape[1]<= 3:
##  Only need 2 points to define a line, so choose two endpoints
        plot_x = [np.amin(X[:,1:2],axis=0)-2,  np.amax(X[:,1:2],axis=0)+2]

##  Calculate the decision boundary line
        list1=[i*theta[1,0] for i in plot_x]
        list1=[j+theta[0,0] - 0.5 for j in list1]
        x=-(1/theta[2,0])
        plot_y = [k*x for k in list1]
        plot_y = np.matrix(plot_y)

##  Plot, and adjust axes for better viewing
        c, =j1.plot(plot_x, plot_y)
        c.set_label('Decision Boundary')
        plt.axis([30,100,30,100])
    else:
## Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))
## Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                qa=mapFeature.mapFeature(u[i], v[j])               
                z[i,j] = qa@theta
        z = z.T ## important to transpose z before calling contour
        c=j1.contour(u,v,z,levels=0)
        c.collections[0].set_label('Decision Boundary')
    return f,j1,a,b;
