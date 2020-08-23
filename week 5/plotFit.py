import numpy as np
import polyFeatures
import matplotlib.pyplot as plt

def func(min_x, max_x, mu, sigma, theta, p):
#PLOTFIT Plots a learned polynomial regression fit over an existing figure.
#Also works with linear regression.
#   PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
#   fit with power p and feature normalization (mu, sigma).


# We plot a range slightly bigger than the min and max values to get
# an idea of how the fit will vary outside the range of the data points
    x=np.array(np.linspace(min_x-15,max_x+25,50))

# Map the X values 
    X_poly = polyFeatures.func(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma

# Add ones
    X_poly = np.concatenate((np.ones((x.shape[0], 1)), X_poly),axis=1)

# Plot
    fig,ax1=plt.subplots()
    plt.plot(x, X_poly @ theta, '--', 'LineWidth', 2)
    return fig,ax1;
