import numpy as np
import featureNormalize
import computeCostMulti
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y) # number of training examples
    J_history = np.zeros((num_iters, 1))
    l,o,n=featureNormalize.featureNormalize(y)
    y=(y-o)/n
    for iter in range (num_iters):
        h=np.dot(X,theta)
        divergence=h-y
        theta=theta-(alpha*(np.dot(X.T,divergence)))/m
        J_history[iter] = computeCostMulti.computeCostMulti(X, y, theta)
    return theta, J_history;
