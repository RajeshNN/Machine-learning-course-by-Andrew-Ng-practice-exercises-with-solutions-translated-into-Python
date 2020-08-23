import numpy as np
import scipy.special as ex

def func(Theta1, Theta2, X):

# Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

# You need to return the following variables correctly 
    p = np.zeros((X.shape[0], 1))

    h1 = ex.expit(np.concatenate((np.ones((m, 1)), X),axis=1) @ Theta1.T)
    h2 = ex.expit(np.concatenate((np.ones((m, 1)), h1),axis=1) @ Theta2.T)
    p = np.argmax(h2,axis=1)+1

    return p;
