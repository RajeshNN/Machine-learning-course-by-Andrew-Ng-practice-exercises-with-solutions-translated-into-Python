import numpy as np
import matplotlib.pyplot as plt
import statistics as st
import math
import computeCostMulti
import featureNormalize
import gradientDescentMulti
import normalEqn

print('Loading data ...\n');

## Load Data
data = np.loadtxt('ex1data2.txt',delimiter=',')
X = data[:, 0:2]
y = data[:, 2:3]
m = len(y)

#Print out some data points
print('First 10 examples from the dataset: \n')
a=X[0:9,:]
b=y[0:9,:]
print('\tX = \t, y = ')
for c in range(len(a)):
    print(a[c], b[c])

# Scale features and set them to zero mean
print('Normalizing Features ...\n')

X, mu, sigma = featureNormalize.featureNormalize(X)

# Add intercept term to X
X = np.concatenate((np.ones((m, 1)), X), axis=1)

print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.5
num_iters = 100

# Init Theta and Run Gradient Descent 
theta1 = np.zeros([3, 1])
theta1, J_history = gradientDescentMulti.gradientDescentMulti(X, y, theta1, alpha, num_iters)

# Plot the convergence graph
plt.plot(range(0,len(J_history)), J_history, '-b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent: \n')
print(theta1)
print('\n')

# Estimate the price of a 1650 sq-ft, 3 br house
#a, mu, sigma=featureNormalize.featureNormalize(data)
l,o,n=featureNormalize.featureNormalize(y)
a1=(1650-mu[0,0])/sigma[0,0]
a2=(3-mu[0,1])/sigma[0,1]
q =np.array([1,a1,a2])
p=q@theta1
price=(p*n)+o

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house(using gradient descent): $', price)

## ================ Part 3: Normal Equations ================

print('Solving with normal equations...\n')

## Load Data
data = np.loadtxt('ex1data2.txt',delimiter=',')
X = data[:, 0:2]
y = data[:, 2:3]
m = len(y)

# Add intercept term to X
X = np.concatenate((np.ones([m, 1]), X), axis=1)

# Calculate the parameters from the normal equation
theta = normalEqn.normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations: \n')
print(' \n', theta)
print('\n')


# Estimate the price of a 1650 sq-ft, 3 br house
price = np.array([1,1650,3])@theta


# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house(using normal equations):\n $', price)

