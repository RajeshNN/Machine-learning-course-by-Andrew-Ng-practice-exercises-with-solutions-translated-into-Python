import scipy.special as ex

def func(z):
    return ex.expit(z)*(1-ex.expit(z));
