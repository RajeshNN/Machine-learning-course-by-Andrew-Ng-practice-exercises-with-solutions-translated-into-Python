from __future__ import division
import numpy as np

def sigmoid(z):
    z=-z
    z=np.exp(z)
    z+=1
    g=np.reciprocal(z)
    return g;
