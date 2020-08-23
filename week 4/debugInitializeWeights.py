import numpy as np
import math

def func(fan_out, fan_in):

# Set W to zeros
    W = np.zeros((fan_out, 1 + fan_in))

# Initialize W using "sin", this ensures that W is always of the same
# values and will be useful for debugging
    x=np.ravel(W)
    
    return np.reshape([math.sin(i) for i in range(len(x))], W.shape) / 10
