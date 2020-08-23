import numpy as np
import scipy.special as ex

def func(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lambdaa):

# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# for our 2 layer neural network\
    a=hidden_layer_size * (input_layer_size + 1)
    
    Theta1 = np.reshape(nn_params[range(a)],(hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[range(a,len(nn_params))],(num_labels, (hidden_layer_size + 1)))

# Setup some useful variables
    m = X.shape[0]
         
    J = 0

    Y=np.zeros((m,num_labels))

    for j in range(m):
        for a in range(1,num_labels+1):
            if y[j,0]==a:
                Y[j,a-1]=1

    A1=np.concatenate((np.ones((m,1)),X),axis=1)
    z2=A1@Theta1.T
    A2=ex.expit(z2)
    A2=np.concatenate((np.ones((A2.shape[0],1)),A2),axis=1)
    z3=A2@Theta2.T
    A3=ex.expit(z3)

    Y1=1-Y
    A3l=np.log(A3.T)
    A31=np.log(1-A3.T)
    a=0
    for i in range(m):
        a=a-(Y[i,:]@A3l[:,i])-(Y1[i,:]@A31[:,i])

    J=a/m
    
    reg=0
    for b1 in range(1,Theta1.shape[1]):
        for a1 in range(Theta1.shape[0]):
            reg+=(Theta1[a1,b1]**2)

    for b2 in range(1,Theta2.shape[1]):
        for a2 in range(Theta2.shape[0]):
            reg+=(Theta2[a2,b2]**2)
  
    J+=((lambdaa*reg)/(2*m))
    J=float(J)

    return J;
