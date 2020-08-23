import numpy as np
import math

def func(x1, x2, sigma):

    x1 = np.ravel(x1)
    x2 = np.ravel(x2)

    sim = 0

    sim=np.exp((-np.sum((x1-x2)**2))/(2*(sigma**2)))

    return sim;
