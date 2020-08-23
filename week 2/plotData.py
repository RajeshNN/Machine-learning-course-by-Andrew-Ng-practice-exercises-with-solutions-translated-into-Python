import numpy as np
import matplotlib.pyplot as plt

def plotData(X,y):
    pos=[]
    neg=[]
    for i in range(len(y)):
        if y[i]==1:
            pos=np.append(pos,i)
        else:
            neg=np.append(neg,i)
    pos=[int(i) for i in pos]
    neg=[int(i) for i in neg]
    fig,ax=plt.subplots()
    a, =ax.plot(X[pos,0],X[pos,1],'+',color='black')
    b, =ax.plot(X[neg,0],X[neg,1],'o',color='yellow',mec='black')
    return fig,ax,a,b;
