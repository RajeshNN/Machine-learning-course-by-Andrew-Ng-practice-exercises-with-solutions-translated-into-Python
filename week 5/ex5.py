import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import linearRegCostFunction
import linearRegGradFunction
import trainLinearReg
import polyFeatures
import learningCurve
import validationCurve
import featureNormalize
import plotFit

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

# Load from ex5data1: 
# You will have X, y, Xval, yval, Xtest, ytest in your environment
data=loadmat ('ex5data1.mat')
X=data['X']
y=data['y']
Xtest=data['Xtest']
ytest=data['ytest']
Xval=data['Xval']
yval=data['yval']

# m = Number of examples
m = X.shape[0]

# Plot training data
plt.plot(X, y, 'x',color='red', MarkerSize= 10, linewidth= 1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear 
#  regression. 
#

theta = np.array([1,1])
J = linearRegCostFunction.func(theta, np.concatenate((np.ones((m, 1)), X),axis=1),\
                               y, 1)

print('Cost at theta = [1  1]:',J,\
      '\n(this value should be about 303.993192)\n')

## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear 
#  regression.
#

theta = np.array([1,1])
grad = linearRegGradFunction.func(theta, np.concatenate((np.ones((m, 1)), X),axis=1),\
                                  y, 1)

print('Gradient at theta = [1  1]:',grad[0], grad[1],\
      '\n(this value should be about [-15.303016 598.250744])\n')

## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train 
#  regularized linear regression.
# 
#  Write Up Note: The data is non-linear, so this will not give a great 
#                 fit.
#

#  Train linear regression with lambdaa = 0
lambdaa = 0
theta = trainLinearReg.func(np.concatenate((np.ones((m, 1)), X),axis=1), y, lambdaa)

#  Plot fit over the data
plt.plot(X, y, 'x',color='red', MarkerSize= 10, LineWidth= 1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.plot(X, np.concatenate((np.ones((m, 1)), X),axis=1)@theta, '--', 'LineWidth', 2)
plt.show()

## =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function. 
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- Figure 3 in ex5.pdf 
#

lambdaa = 0
error_train, error_val =\
             learningCurve.func(np.concatenate((np.ones((m, 1)), X),axis=1), y,\
                                np.concatenate((np.ones((Xval.shape[0], 1)), Xval),axis=1),\
                                yval,lambdaa)

fig,ax=plt.subplots()
l1,=plt.plot(range(m), error_train)
l2,=plt.plot(range(m), error_val)
plt.title('Learning curve for linear regression')
plt.legend((l1,l2),('Train', 'Cross Validation'))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
xmin,xmax,ymin,ymax=plt.axis([0, 13, 0, 150])
plt.show()

print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('\t', i+1,'\t\t',error_train[i],'\t',error_val[i],'\n')

## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures.func(X, p)
X_poly, mu, sigma = featureNormalize.func(X_poly)  # Normalize
X_poly = np.concatenate((np.ones((m, 1)), X_poly),axis=1)                   # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures.func(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.concatenate((np.ones((X_poly_test.shape[0],1)), X_poly_test),axis=1)# Add Ones

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures.func(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.concatenate((np.ones((X_poly_val.shape[0],1)), X_poly_val),axis=1)   # Add Ones

print('Normalized Training Example 1:\n')
print('\n', X_poly[1, :],'\n')

## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambdaa. The code below runs polynomial regression with 
#  lambdaa = 0. You should try running the code with different values of
#  lambdaa to see how the fit and learning curve change.
#

lambdaa = 3
theta = trainLinearReg.func(X_poly, y, lambdaa)

# Plot training data and fit
fig1,ax1=plotFit.func(min(X), max(X), mu, sigma, theta, p)
l,=plt.plot(X, y, 'x',color='red', MarkerSize= 10)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title ('Polynomial Regression Fit (lambdaa =3)')

fig2,ax2=plt.subplots()
error_train, error_val =learningCurve.func(X_poly, y, X_poly_val, yval, lambdaa)
l1,=plt.plot(range(m), error_train)
l2,=plt.plot(range(m), error_val)

plt.title('Polynomial Regression Learning Curve (lambdaa = 3)')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
xmin,xmax,ymin,ymax= plt.axis([0, 13, 0, 100])
plt.legend((l1,l2),('Train', 'Cross Validation'))
fig1.show()
fig2.show()

print('Polynomial Regression (lambdaa =', lambdaa,')')
print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('\t', i+1,'\t\t', error_train[i],'\t', error_val[i],'\n')

## =========== Part 8: Validation for Selecting lambdaa =============
#  You will now implement validationCurve to test various values of 
#  lambdaa on a validation set. You will then use this to select the
#  "best" lambdaa value.
#

lambdaa_vec, error_train, error_val =\
             validationCurve.func(X_poly, y, X_poly_val, yval)

fig3=plt.figure()
l3,=plt.plot(lambdaa_vec, error_train)
l4,=plt.plot(lambdaa_vec, error_val)
plt.legend((l3,l4),('Train', 'Cross Validation'))
plt.xlabel('lambdaa')
plt.ylabel('Error')
plt.show()

print('lambdaa\t\tTrain Error\tValidation Error\n')
for i in range(len(lambdaa_vec)):
    print(lambdaa_vec[i],'\t', error_train[i],'\t', error_val[i])
