from scipy.io import loadmat
import scipy.optimize as sc
import numpy as np
import matplotlib.pyplot as plt
import displayData
import lrCostFunc
import lrgradFunc
import oneVsAll
import predictOneVsAll


## Setup the parameters you will use for this part of the exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
num_labels = 10          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

data=loadmat('ex3data1.mat') # training data stored in arrays X, y
X=data['X']
X=np.array(X)
y=data['y']
y=np.matrix(y)
m = X.shape[0]

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = np.matrix(X[rand_indices[0:100], :])

displayData.displD(sel)


## ============ Part 2a: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#

# Test case for lrCostFunction
print('\nTesting lrCostFunction() with regularization')

theta_t = np.array([-2,-1,1,2])
a=np.reshape(range(1,16),(5,3),order='F')/10
X_t = np.concatenate((np.ones((5,1)), a),axis=1)
y_t = np.matrix([[1],[0],[1],[0],[1]])
lambda_t = 3
J = lrCostFunc.lrCostFunction(theta_t, X_t, y_t, lambda_t)
grad= lrgradFunc.lrgradFunc(theta_t,X_t,y_t,lambda_t)

print('\nCost:', J)
print('Expected cost: 2.534819\n')
print('Gradients:\n')
print(grad)
print('Expected gradients:\n')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')

## ============ Part 2b: One-vs-All Training ============
print('\nTraining One-vs-All Logistic Regression...\n')

lambdaa = 0.01
all_theta = oneVsAll.oneVsAll(X, y, num_labels, lambdaa)


## ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll.predictOVA(all_theta, X)
r=0
for i in range(y.shape[0]):
    if pred[i]==y[i]:
        r+=1
r=r/(y.shape[0])

print('\nTraining Set Accuracy:', r * 100)

