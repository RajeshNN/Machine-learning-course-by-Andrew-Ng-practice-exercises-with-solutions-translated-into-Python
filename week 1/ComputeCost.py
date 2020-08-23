from __future__ import division
import numpy as np
import math
def ComputeCost(X,y,theta):
  m=X.shape[0]
  J=0
  h=X@theta
  div=h-y
  div=div**2
  p=np.sum(div)
  J=p/(2*m)
  return J;
