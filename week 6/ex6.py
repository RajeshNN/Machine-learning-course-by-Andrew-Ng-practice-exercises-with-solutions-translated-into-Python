import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import plotData
from sklearn import svm
import svmTrain
import visualizeBoundary
import gaussianKernel
import gaussianKernelGramMatrix
import dataset3Params

## =============== Part 1: Loading and Visualizing Data ================
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

print('Loading and Visualizing Data ...\n')

# Load from ex6data1: 
# You will have X, y in your environment
data1=loadmat('ex6data1.mat')
X=data1['X']
y=data1['y']

# Plot training data
plotData.func(X, y)
plt.show()

## ==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.
#

# Load from ex6data1: 
# You will have X, y in your environment
data1=loadmat('ex6data1.mat')
X=data1['X']
y=data1['y']

print('\nTraining Linear SVM ...\n')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 1
model = svmTrain.func(X,y,C,'linear')
fig1,ax1,a,b=visualizeBoundary.func(X,y,model)
fig1.show()

## =============== Part 3: Implementing Gaussian Kernel ===============
#  You will now implement the Gaussian kernel to use
#  with the SVM. You should complete the code in gaussianKernel.m
#
print('\nEvaluating the Gaussian Kernel ...\n')

x1 = np.array([[1],[2],[1]])
x2 = np.array([[0],[4],[-1]])
sigma = 2
sim = gaussianKernel.func(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma =',\
      sigma,':',sim,'\n\n(for sigma = 2, this value should be about 0.324652)\n')

## =============== Part 4: Visualizing Dataset 2 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
#

print('Loading and Visualizing Data ...\n')

# Load from ex6data2: 
# You will have X, y in your environment
data2=loadmat('ex6data2.mat')
X=data2['X']
y=data2['y']

# Plot training data
plotData.func(X, y)
plt.show()

## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the 
#  SVM classifier.
# 
print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n')

# Load from ex6data2: 
# You will have X, y in your environment
data2=loadmat('ex6data2.mat')
X=data2['X']
y=data2['y']

# SVM Parameters
C = 1
sigma = 0.1

# We set the tolerance and max_passes lower here so that the code will run
# faster. However, in practice, you will want to run the training to
# convergence.

model= svmTrain.func(X,np.array(y),C,'gaussian',sigma)
fig2,ax2,a2,b2=visualizeBoundary.func(X,y, model)
fig2.show

## =============== Part 6: Visualizing Dataset 3 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
#

print('Loading and Visualizing Data ...\n')

# Load from ex6data3: 
# You will have X, y in your environment
data3=loadmat('ex6data3.mat')
X=data3['X']
y=data3['y']

# Plot training data
plotData.func(X, y)
plt.show()

## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

#  This is a different dataset that you can use to experiment with. Try
#  different values of C and sigma here.
# 

# Load from ex6data3: 
# You will have X, y in your environment
data3=loadmat('ex6data3.mat')
X=data3['X']
y=data3['y']
Xval=data3['Xval']
yval=data3['yval']

# Try different SVM Parameters here
C, sigma = dataset3Params.func(X, y, Xval, yval)

# Train the SVM
model= svmTrain.func(X,y,C,'gaussian',sigma)
fig3,ax3,a3,b3=visualizeBoundary.func(X,y, model)
fig3.show()

