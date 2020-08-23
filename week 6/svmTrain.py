import numpy as np
from sklearn import svm
import gaussianKernel
import gaussianKernelGramMatrix

def func(X, y, C, kernelFunction='gaussian',sig=0.3):

    if kernelFunction=='linear':
        model=svm.SVC(kernel='linear')
        model.fit(X,np.ravel(y))
        return model;
    else:
        model=svm.SVC(kernel='precomputed')
        model.fit(gaussianKernelGramMatrix.func(X,X,sig),np.ravel(y))
        return model;
