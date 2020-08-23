from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
#import imagesc
import math

def displD(X):
    
    example_width=round(math.sqrt(X.shape[1]))

# Compute rows, cols
    [m, n] = X.shape
    example_height = (n / example_width)
    example_height =int(example_height)
# Compute number of items to display
    display_rows = math.floor(math.sqrt(m))
    display_cols = math.ceil(m / display_rows)
# Between images padding
    pad = 1

# Setup blank display
    display_array =  np.ones((pad + display_rows * (example_height + pad), \
                       pad + display_cols * (example_width + pad)))

# Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break
            max_val = max(abs(X[curr_ex, :]))
            a=pad + (j * (example_height + pad))
            b=pad + (i * (example_width + pad))
            display_array[a:(a + example_height), b:(b + example_width)] = \
                          np.reshape(X[curr_ex, :], (example_height, example_width),order='F')
            curr_ex = curr_ex + 1
        if curr_ex>m :
            break

# Display Image
    h = plt.imshow(display_array, cmap='gray_r')
    plt.axis('off')
    plt.show()

