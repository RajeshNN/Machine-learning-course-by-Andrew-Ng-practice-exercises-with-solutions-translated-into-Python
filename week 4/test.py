import numpy as np
from scipy.io import loadmat
import nnCostAndGrad
import nnCostFunction
import sigmoidGradient
import randInitializeWeights
import fmincg

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)
# Load Training Data
print('Loading and Visualizing Data ...\n')

data=loadmat('ex4data1.mat')
X=np.array(data['X'],dtype=np.float32)
y=np.array(data['y'],dtype=np.float32)
m = X.shape[0]

print('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights.func(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights.func(hidden_layer_size, num_labels)

# Unroll parameters

initial_nn_params = np.matrix(np.append(np.ravel(initial_Theta1),np.ravel(initial_Theta2)))
initial_nn_params=np.array(initial_nn_params.T)
print('\nTraining Neural Network... \n')

#  You should also try different values of lambdaa
lambdaa = 1

# Now, costFunction is a function that takes in only one argument (the
# neural network parameters)

def costfunc(ils,hls,nl,u,v,l):
        return lambda t: nnCostAndGrad.func(t,ils, hls,nl, u, v, l);

CnG=costfunc(input_layer_size,hidden_layer_size,num_labels,X,y,lambdaa)

nn_params = fmincg.func(CnG,initial_nn_params)

cost= nnCostFunction.func(nn_params, input_layer_size,\
                          hidden_layer_size, num_labels, X, y, lambdaa)
print('cost:',cost)
