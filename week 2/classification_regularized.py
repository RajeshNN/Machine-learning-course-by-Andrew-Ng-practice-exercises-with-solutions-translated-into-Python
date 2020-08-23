from __future__ import division
import numpy as np
import scipy.optimize as sc
import matplotlib.pyplot as plt
import sigmoid
import mapFeature
import costFunctionReg
import gradfunReg
import predict
import plotData
import pdb

data = np.loadtxt('ex2data2.txt',delimiter=',')
X = data[:, 0:2]
y = data[:, 2:3]

f,j1,a,b=plotData.plotData(X, y)

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

a.set_label('y=1')
b.set_label('y=0')
j1.legend()
plt.show()


## =========== Part 1: Regularized Logistic Regression ============


# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mapFeature.mapFeature(X[:,0:1], X[:,1:2])

# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1))

# Set regularization parameter lambdaa to 1
lambdaa = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost = costFunctionReg.costFunctionReg(initial_theta, X, y, lambdaa)
grad = gradfunReg.gradfunReg(initial_theta, X, y, lambdaa)
grad=np.matrix(grad)
print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:\n')
print(' \n', grad[0:5,0])
print('Expected gradients (approx) - first five values only:\n')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')


# Compute and display cost and gradient
# with all-ones theta and lambdaa = 10
test_theta = np.ones((X.shape[1],1))
cost = costFunctionReg.costFunctionReg(test_theta, X, y, 10)
grad = gradfunReg.gradfunReg(test_theta, X, y, 10)
grad=np.matrix(grad)
print('\nCost at test theta (with lambda = 10): ', cost)
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:\n')
print('\n', grad[0:5,0])
print('Expected gradients (approx) - first five values only:\n')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')


## ============= Part 2: Regularization and Accuracies =============



# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1))

# Set regularization parameter lambdaa to 1 (you should vary this)
lambdaa = 1

# Optimize
res = sc.minimize(costFunctionReg.costFunctionReg, initial_theta, args=(X,y,lambdaa),)
theta=np.matrix(res.x).T
print(res.message)
cost=costFunctionReg.costFunctionReg(theta,X,y,lambdaa)

# Plot Boundary
f,j1,a,b=pdb.plotDecisionBoundary(theta, X, y)

plt.title('lambda =1')

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

a.set_label('y=1')
b.set_label('y=0')
j1.legend()
plt.show()

# Compute accuracy on our training set
p = predict.predict(theta, X)
q=np.zeros((p.shape))
for i in range(len(p)):
        if p[i]==y[i]:
                q[i]=1
r=np.mean(q)

print('Train Accuracy:', r * 100)
print('Expected accuracy (with lambda = 1): 83.1 (approx)\n')


