import numpy as np
import scipy.special as ex

def predict(Theta1, Theta2, X):

    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

# You need to return the following variables correctly
    p = np.zeros((m, 1))

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a 
#               vector containing labels between 1 to num_labels.
#
# Hint: The max function might come in useful. In particular, the max
#       function can also return the index of the max element, for more
#       information see 'help max'. If your examples are in rows, then, you
#       can use max(A, [], 2) to obtain the max for each row.
#

    X=np.concatenate((np.ones((m,1)), X),axis=1)
    z2=X@Theta1.T
    a2=ex.expit(z2)
    a2=np.concatenate((np.ones((a2.shape[0],1)), a2),axis=1)
    z3=a2@Theta2.T
    a3=ex.expit(z3)
    p=np.argmax(a3,axis=1)
    return p+1;
