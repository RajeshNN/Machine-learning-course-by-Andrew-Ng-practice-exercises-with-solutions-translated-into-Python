import ComputeCost
import numpy as np
def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y) # number of training examples
    J_history = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        h=X@theta
        divergence=h-y
       # l=divergence.T@X[:,0:2]
        temp1=theta[0,0]-(alpha*(divergence.T@X[:,0]))/m
        temp2=theta[1,0]-(alpha*(divergence.T@X[:,1]))/m
        theta[0,0]=temp1
        theta[1,0]=temp2   
        J_history[iter] = ComputeCost.ComputeCost(X, y, theta)
        iter+=1
    return theta, J_history;
