import numpy as np

def func(L_in, L_out):

    e = 0.12
    return ((np.random.randint(0,2000,size=(L_out,L_in+1))/1000)*e)-e;
