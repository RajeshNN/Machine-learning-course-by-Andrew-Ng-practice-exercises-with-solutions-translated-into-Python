from __future__ import division
import numpy as np
import scipy.optimize as sc
import matplotlib.pyplot as plt
import sigmoid
import costFunction
import gradfun
import predict
import plotData
import pdb


## Load Data
## The first two columns contains the exam scores and the third column
## contains the label.

data = np.loadtxt('ex2data1.txt',delimiter=',')
X = data[:,0:2]
y = np.array(data[:,2:3])

## ==================== Part 1: Plotting ====================
##  We start the exercise by first plotting the data to understand the 
##  the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')

f,j1,a,b=plotData.plotData(X,y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
a.set_label('y=1')
b.set_label('y=0')
j1.legend()
plt.show()

# Specified in plot order
plt.show()

##%% ============ Part 2: Compute Cost and Gradient ============
##%  In this part of the exercise, you will implement the cost and gradient
##%  for logistic regression. You neeed to complete the code in 
##%  costFunction.m

##%  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

##% Add intercept term to x and X_test
X = np.concatenate((np.ones((m, 1)), X),axis=1)

##% Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1))

##% Compute and display initial cost and gradient
cost= costFunction.costFunction(initial_theta, X, y)
grad= gradfun.gradfun(initial_theta,X,y)

print('Cost at initial theta (zeros): ', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): ')
print(grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

##% Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24],[0.2],[0.2]])
cost= costFunction.costFunction(test_theta, X, y)
grad= gradfun.gradfun(test_theta,X,y)

print('\nCost at test theta: ', cost)
print('\nExpected cost (approx): 0.218\n')
print('Gradient at test theta: \n')
print('\n', grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')


##%% ============= Part 3: Optimizing using fminunc  =============
##%  In this exercise, you will use a built-in function (fminunc) to find the
##%  optimal parameters theta.

##%  This function will return theta and the cost

res = sc.minimize(costFunction.costFunction, [initial_theta], args=(X,y), method='Nelder-Mead')
theta=np.matrix(res.x).T
print(res.message)
cost=costFunction.costFunction(theta,X,y)
grad=gradfun.gradfun(theta,X,y)
##% Print theta to screen
print('Cost at theta: ', cost)
print('\nExpected cost (approx): 0.203\n')
print('theta: \n')
print('\n', theta)
print('\nExpected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')

##% Plot Boundary
f,j1,a,b=pdb.plotDecisionBoundary(theta, X, y)

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
a.set_label('y=1')
b.set_label('y=0')
j1.legend(loc='upper right')
plt.show()

##%% ============== Part 4: Predict and Accuracies ==============
##%  After learning the parameters, you'll like to use it to predict the outcomes
##%  on unseen data. In this part, you will use the logistic regression model
##%  to predict the probability that a student with score 45 on exam 1 and 
##%  score 85 on exam 2 will be admitted.
##%
##%  Furthermore, you will compute the training and test set accuracies of 
##%  our model.
##%
##%  Your task is to complete the code in predict.m
##
##%  Predict probability for a student with score 45 on exam 1 
##%  and score 85 on exam 2 

prob = sigmoid.sigmoid([1, 45, 85] * theta)
print('For a student with scores 45 and 85, we predict an admission probability of', prob)
print('\nExpected value: 0.775 +/- 0.002\n\n')

##% Compute accuracy on our training set
p = predict.predict(theta, X)
q=np.zeros((p.shape))
for i in range(len(p)):
        if p[i]==y[i]:
                q[i]=1
r=np.mean(q)
r*=100

print('Train Accuracy:', r)
print('Expected accuracy (approx): 89.0\n')
print('\n')


