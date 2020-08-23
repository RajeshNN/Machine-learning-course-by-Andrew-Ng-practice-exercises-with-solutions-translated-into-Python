from __future__ import division
from mpl_toolkits import mplot3d
import ComputeCost
import gradientDescent
import numpy as np
import matplotlib.pyplot as plt

#plotting data
data=np.loadtxt("ex1data1.txt",delimiter=',')
theta = np.zeros([2, 1])
X=data[:,0:1]
y=data[:,1:2]
m=y.shape[0]
plt.plot(X,y,'rx')
plt.xlabel("Population of cities in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.show()

#testing cost function
print('\nTesting the cost function ...\n...\n...\n')
X=np.concatenate((np.ones([data.shape[0],1]),data[:,0:1]),axis=1)
J = ComputeCost.ComputeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed =', J)
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = ComputeCost.ComputeCost(X, y, np.array([[-1 ],[ 2]]))
print('\nWith theta = [-1 ; 2]\nCost computed =', J)
print('Expected cost value (approx) 54.24\n')

#running gradient descent
alpha=0.005
iterations=3000
theta,J_history = gradientDescent.gradientDescent(X, y, theta, alpha, iterations);

#print theta to screen
print('Theta found by gradient descent:\n')
print('\n', theta)
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

#Plot the linear fit
plt.plot(X[:,1:2],y,'rx')
plt.plot(X[:,1:2], X@theta)
plt.xlabel("Population of cities in 10,000s")
plt.ylabel("Profit in $10,000s")
#plt.legend('Training data', 'Linear regression')
plt.show()

#Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]) @theta
print('For population = 35,000, we predict a profit of \n',predict1*10000)
predict2 = np.array([1, 7]) @ theta
print('For population = 70,000, we predict a profit of \n',predict2*10000)

#visualizing J
print('Visualizing J(theta_0, theta_1) ...\n')

#Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, num=100)
theta1_vals = np.linspace(-1, 4, num=100)

#initialize J_vals to a matrix of 0's
J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])

#Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        a1=theta0_vals[i]
        a2=theta1_vals[j]
        t = np.array([[a1], [a2]])
        J_vals[i,j] = ComputeCost.ComputeCost(X, y, t)



### Because of the way meshgrids work in the surf command, we need to
### transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T

###Surface plot
fig = plt.figure()
ax= plt.axes(projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals,cmap='viridis', edgecolor='none')
plt.xlabel('theta_0'); plt.ylabel('theta_1');
plt.show()

# Contour plot

# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
plt.contour(theta0_vals, theta1_vals, J_vals, levels=np.logspace(-2, 3))
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.plot(theta[0], theta[1], 'rx')
plt.show()
