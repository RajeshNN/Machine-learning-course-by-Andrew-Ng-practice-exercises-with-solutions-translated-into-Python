import numpy as np
import scipy.special as ex
import sigmoidGradient

def func(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lambdaa):

    a=hidden_layer_size * (input_layer_size + 1)

    Theta1 = np.reshape(nn_params[range(a)],(hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[range(a,len(nn_params))],(num_labels, (hidden_layer_size + 1)))

# Setup some useful variables
    m = X.shape[0]
         
# You need to return the following variables correctly 
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    Y=np.zeros((m,num_labels))
    delta1=0
    delta2=0
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

    d3=A3-Y
    d2=np.multiply((d3@Theta2[:,range(1,Theta2.shape[1])]),sigmoidGradient.func(z2))
    delta2=d3.T@A2
    delta1=d2.T@A1
    Theta1_grad=delta1/m
    Theta2_grad=delta2/m
    for p in range(1,Theta1_grad.shape[1]):
        Theta1_grad[:,p]+=((lambdaa/m)*Theta1[:,p])

    for q in range(1,Theta2_grad.shape[1]):
        Theta2_grad[:,q]+=((lambdaa/m)*Theta2[:,q])

# Unroll gradients
    grad = np.append(np.ravel(Theta1_grad),np.ravel(Theta2_grad))

    return grad;
