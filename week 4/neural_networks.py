import numpy as np
from scipy.io import loadmat
import scipy.optimize as sc
import matplotlib.pyplot as plt
import displayData
import nnCostFunction
import nnGradFunction
import sigmoidGradient
import randInitializeWeights
import checkNNGradients
import predict

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

data=loadmat('ex4data1.mat')
X=data['X']
y=data['y']
m = X.shape[0]

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = rand_indices[1:100]

displayData.displD(np.matrix(X[sel, :]))


## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
weights=loadmat('ex4weights.mat')
Theta1=weights['Theta1']
Theta2=weights['Theta2']

# Unroll parameters 
nn_params = np.append(np.ravel(Theta1),np.ravel(Theta2))

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('\nFeedforward Using Neural Network ...\n')

# Weight regularization parameter (we set this to 0 here).
lambdaa = 0

J = nnCostFunction.func(nn_params, input_layer_size, hidden_layer_size,\
                        num_labels, X, y, lambdaa)

print('Cost at parameters (loaded from ex4weights):',J,\
      '\n(this value should be about 0.287629)\n')

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('\nChecking Cost Function (w/ Regularization) ... \n')

# Weight regularization parameter (we set this to 1 here).
lambdaa = 1

J = nnCostFunction.func(nn_params, input_layer_size, hidden_layer_size, \
                        num_labels, X, y, lambdaa)

print('Cost at parameters (loaded from ex4weights):',J,\
      '\n(this value should be about 0.383770)\n')


## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient.func(np.array([[-1],[-0.5],[0],[0.5],[1]]))
print('Sigmoid gradient evaluated at [-1;-0.5;0;0.5;1]:\n')
print(g)
print('\n\n')

## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights.func(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights.func(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.append(np.ravel(initial_Theta1),np.ravel(initial_Theta2))

## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#
print('\nChecking Backpropagation... \n')

#  Check gradients by running checkNNGradients
checkNNGradients.func()

## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

print('\nChecking Backpropagation (w/ Regularization) ... \n')

#  Check gradients by running checkNNGradients
lambdaa = 3
checkNNGradients.func(lambdaa)

# Also output the costFunction debugging values
debug_J  = nnCostFunction.func(nn_params, input_layer_size, \
                               hidden_layer_size, num_labels, X, y, lambdaa)

print('\n\nCost at (fixed) debugging parameters (w/ lambdaa =',lambdaa,'):', debug_J , \
      '\n(for lambdaa = 3, this value should be about 0.576051)\n\n')


## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural 
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print('\nTraining Neural Network... \n')

#  You should also try different values of lambdaa
lambdaa = 1

# Now, costFunction is a function that takes in only one argument (the
# neural network parameters)
nn_params = sc.fmin_cg(nnCostFunction.func,initial_nn_params,\
                  args=(input_layer_size,hidden_layer_size,\
                        num_labels,X,y,lambdaa),\
                  fprime=nnGradFunction.func,maxiter=50)

cost= nnCostFunction.func(nn_params, input_layer_size,\
                          hidden_layer_size, num_labels, X, y, lambdaa)


# Obtain Theta1 and Theta2 back from nn_params
a=hidden_layer_size * (input_layer_size + 1)
Theta1 = np.reshape(nn_params[range(a)],(hidden_layer_size, (input_layer_size + 1)))

Theta2 = np.reshape(nn_params[range(a,len(nn_params))],(num_labels, (hidden_layer_size + 1)))


## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by 
#  displaying the hidden units to see what features they are capturing in 
#  the data.

print('\nVisualizing Neural Network... \n')

displayData.displD(Theta1[:, range(1,Theta1.shape[1])])

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict.func(Theta1, Theta2, X)
r=0
for i in range(m):
    if pred[i]==y[i,0]:
        r+=1    
r=r/m
print('\nTraining Set Accuracy:', r * 100)


