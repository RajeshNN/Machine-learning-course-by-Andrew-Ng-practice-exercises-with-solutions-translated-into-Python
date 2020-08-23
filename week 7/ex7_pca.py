import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import featureNormalize
import drawLine
import projectData
import recoverData
import pca
import displayData
import kMeansInitCentroids
import runkMeans
import plotDataPoints

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize
#
print('Visualizing example dataset for PCA.\n\n')

#  The following command loads the dataset. You should now have the 
#  variable X in your environment
data= loadmat('ex7data1.mat')
X=data['X']

#  Visualize the example dataset
fig1,ax1=plt.subplots()
ax1.plot(X[:, 0], X[:, 1], 'bo')
xmin,xmax,ymin,ymax=plt.axis([0.5,6.5,2,8])
plt.show()

## =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
#
print('\nRunning PCA on example dataset.\n\n')

#  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = featureNormalize.func(X)

#  Run PCA
U, S = pca.func(X_norm)

#  Compute mu, the mean of the each feature

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.
fig2,ax2=plt.subplots()
ax2.plot(X[:, 0], X[:, 1], 'bo')
xmin,xmax,ymin,ymax=plt.axis([0.5,6.5,2,8])

ax2=drawLine.func(mu.T, mu.T + 1.5 * S[0] * np.array(np.matrix(U[:,0])).T, ax2)
ax2=drawLine.func(mu.T, mu.T + 1.5 * S[1] * np.array(np.matrix(U[:,0])).T, ax2)
plt.show()
print('Top eigenvector: \n')
print(' U[:,0] =', U[0,0],',', U[1,0],'\n')
print('\n(you should expect to see -0.707107 -0.707107)\n')


## =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the 
#  first k eigenvectors. The code will then plot the data in this reduced 
#  dimensional space.  This will show you what the data looks like when 
#  using only the corresponding eigenvectors to reconstruct it.
#
#  You should complete the code in projectData.m
#
print('\nDimension reduction on example dataset.\n\n')

fig3,ax3=plt.subplots()

#  Plot the normalized dataset (returned rom pca)
plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
xmin,xmax,ymin,ymax=plt.axis([-4,3,-4,3])

#  Project the data onto K = 1 dimension
K = 1
Z = projectData.func(X_norm, U, K)
print('Projection of the first example: %f\n'%Z[0])
print('\n(this value should be about 1.481274)\n\n')

X_rec  = recoverData.func(Z, U, K)
print('Approximation of the first example:', X_rec[0,0],',', X_rec[0,1],'\n')
print('\n(this value should be about  -1.047419 -1.047419)\n\n')

#  Draw lines connecting the projected points to the original points

ax3.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
for i in range(X_norm.shape[0]):
    ax3=drawLine.func(X_norm[i,:], X_rec[i,:],ax3)
plt.show()

## =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
print('\nLoading face dataset.\n\n')

#  Load Face dataset
faces=loadmat('ex7faces.mat')
X=faces['X']

#  Display the first 100 faces in the dataset
displayData.displD(X[:100, :])

## =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
#
print('\nRunning PCA on face dataset.\n'\
       '(this might take a minute or two ...)\n\n')

#  Before running PCA, it is important to first normalize X by subtracting 
#  the mean value from each feature
X_norm, mu, sigma = featureNormalize.func(X)

#  Run PCA
U, S = pca.func(X_norm)

#  Visualize the top 36 eigenvectors found
displayData.displD(U[:, :36].T)

## ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors 
#  If you are applying a machine learning algorithm 
print('\nDimension reduction for face dataset.\n\n')

K = 100
Z = projectData.func(X_norm, U, K)

print('The projected data Z has a size of: ')
print(Z.shape)

## ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and 
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed

print('\nVisualizing the projected (reduced dimension) faces.\n\n')

K = 100
X_rec  = recoverData.func(Z, U, K)

# Display normalized data
fig4=plt.figure()
ax4=fig4.add_subplot(1, 2, 1)
ax4=displayData.displD(X_norm[:100,:],ax4,fig4)
plt.title('Original faces')

# Display reconstructed data from only k eigenfaces
ax5=fig4.add_subplot(1, 2, 2)
ax5=displayData.displD(X_rec[:100,:],ax5,fig4)
plt.title('Recovered faces')
plt.show()

## === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
#  One useful application of PCA is to use it to visualize high-dimensional
#  data. In the last K-Means exercise you ran K-Means on 3-dimensional 
#  pixel colors of an image. We first visualize this output in 3D, and then
#  apply PCA to obtain a visualization in 2D.

# Reload the image from the previous exercise and run K-Means on it
# For this to work, you need to complete the K-Means assignment first
A = plt.imread('bird_small.png')

# If imread does not work for you, you can try instead
#   load ('bird_small.mat')

img_size = A.shape
X = np.reshape(A, (img_size[0] * img_size[1], 3))
K = 16 
max_iters = 10
initial_centroids = kMeansInitCentroids.func(X, K)
centroids, idx = runkMeans.func(X, initial_centroids, max_iters)

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
sel = np.floor(np.random.rand(1000, 1) * X.shape[0])

#  Setup Color Palette
palette = np.random.rand(K,3)
#colors=np.zeros((sel.shape[0],palette.shape[1]))

figg = plt.figure()
axx = figg.add_subplot(111, projection='3d')

for i in range(sel.shape[0]):
    colors = palette[int(idx[int(sel[i])]), :]
    #  Visualize the data and centroid memberships in 3D
    axx.scatter(X[int(sel[i]), 0], X[int(sel[i]), 1], X[int(sel[i]), 2], color=colors)
plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
plt.show()

## === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization

# Subtract the mean to use PCA
X_norm, mu, sigma = featureNormalize.func(X)

# PCA and project the data to 2D
U, S = pca.func(X_norm)
Z = projectData.func(X_norm, U, 2)

Z1=np.zeros((sel.shape[0],Z.shape[1]))
idx1=np.zeros((sel.shape[0],idx.shape[1]))

for j in range(sel.shape[0]):
    Z1[j,:]=Z[int(sel[j]),:]
    idx1[j]=idx[int(sel[j])]

# Plot in 2D
figg2=plt.figure()
axx2=figg2.add_subplot()
figg2,axx2=plotDataPoints.func(Z1,idx1, K, axx2, figg2)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
plt.show()
