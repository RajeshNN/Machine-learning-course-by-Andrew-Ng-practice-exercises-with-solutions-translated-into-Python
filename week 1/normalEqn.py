import numpy as np
def normalEqn(X, y):
    theta = np.zeros((X.shape[1], 1))
    theta=(np.linalg.inv(X.T@X))@(X.T@y)
    return theta;
