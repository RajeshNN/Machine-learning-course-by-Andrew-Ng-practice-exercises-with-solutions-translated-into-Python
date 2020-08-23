import numpy as np
import svmTrain
import gaussianKernelGramMatrix

def func(X, y, Xval, yval):
#DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
#where you select the optimal (C, sigma) learning parameters to use for SVM
#with RBF kernel
#   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
#   sigma. You should complete this function to return the optimal C and 
#   sigma based on a cross-validation set.
#

# You need to return the following variables correctly.
    C = 1
    sigma = 0.3
    pred= np.zeros((Xval.shape[0]))
    a=[0.01,0.03,0.1,0.3,1,3,10,30]
    d=np.zeros((8,8))
    for i in range(len(a)):
        print('iteration set',(i+1),'/8')
        for j in range(len(a)):
            model= svmTrain.func(X, y, a[i], 'gaussian', sig=a[j])
            model.fit(gaussianKernelGramMatrix.func(X,X,a[j]),np.ravel(y))
            pred=model.predict(gaussianKernelGramMatrix.func(Xval,X,a[j]))
            d[i,j]=np.mean(pred != yval)
    x=np.amin(d,axis=0)
    xi=np.argmin(d,axis=0)
    yi=np.argmin(x)
    p=xi[yi]
    q=yi
    C=a[p]
    sigma=a[q]
    return C,sigma;
