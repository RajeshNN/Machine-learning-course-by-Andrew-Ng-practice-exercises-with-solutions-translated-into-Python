import numpy as np
import nnGradFunction
import nnCostFunction
import debugInitializeWeights
import computeNumericalGradient

def func (lambdaa=0):
#CHECKNNGRADIENTS Creates a small neural network to check the
#backpropagation gradients
#   CHECKNNGRADIENTS(lambdaa) Creates a small neural network to check the
#   backpropagation gradients, it will output the analytical gradients
#   produced by your backprop code and the numerical gradients (computed
#   using computeNumericalGradient). These two gradient computations should
#   result in very similar values.
#

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

# We generate some 'random' test data
    Theta1 = debugInitializeWeights.func(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights.func(num_labels, hidden_layer_size)
# Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights.func(m, input_layer_size - 1)
    y  = (np.matrix([i%num_labels for i in range(m)])+2).T
    y  = np.array(y)

# Unroll parameters
    nn_params = np.append(np.ravel(Theta1),np.ravel(Theta2))

    def costfunc(ils,hls,nl,u,v,l):
        return lambda t: nnCostFunction.func(t,ils, hls,nl, u, v, l);

    cost=costfunc(input_layer_size, hidden_layer_size,num_labels, X, y, lambdaa)

    grad = nnGradFunction.func(nn_params,input_layer_size, hidden_layer_size,\
                              num_labels, X, y, lambdaa)
    
    numgrad = computeNumericalGradient.func(cost, nn_params)

# Visually examine the two gradient computations.  The two columns
# you get should be very similar.
    q=np.zeros((grad.shape[0],2))
    q[:,0]=numgrad
    q[:,1]=grad
    print(q)
    print('The above two columns you get should be very similar.\n' \
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')

# Evaluate the norm of the difference between two solutions.  
# If you have a correct implementation, and assuming you used EPSILON = 0.0001 
# in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

    print('If your backpropagation implementation is correct, then \n' \
          'the relative difference will be small (less than 1e-9). \n' \
          '\nRelative Difference:', diff)
