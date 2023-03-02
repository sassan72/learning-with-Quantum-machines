# Gaussian Process with quantum kernel

import numpy as np

from sklearn.svm import SVC

import pennylane as qml
import matplotlib.pyplot as plt
import sklearn.decomposition
from sklearn import metrics
from scipy.linalg import solve

import math

#data preprocessing 

data1 = np.loadtxt("TDS.txt") # import pandas as pd

data2 = np.loadtxt("TLD.txt") #

data3 = np.loadtxt("VDS.txt") #

data4 = np.loadtxt("VLD.txt") #

y_train= data2[:,] 

X_train2=data1[:]

y_test= data4[:,] 

X_test2=data3[:]


n_qubits = 3

pi = np.pi


X_train1 = sklearn.preprocessing.normalize(X_train2, norm='l2',axis=0)

X_test1 = sklearn.preprocessing.normalize(X_test2, norm='l2', axis=0)

X_train = sklearn.preprocessing.normalize(X_train1, norm='l2',axis=1)

X_test = sklearn.preprocessing.normalize(X_test1, norm='l2', axis=1)

y_train_new =[]


for i in range(len(X_train)):
    if y_train[i]==1:
        a=1.
        y_train_new.append(a)
    else:
        b=-1.
        y_train_new.append(b)


######################################################################
# Build the quantum kernel function

dev_kernel = qml.device("default.qubit", wires=n_qubits)


projector = np.zeros((2**n_qubits, 2**n_qubits))
projector[0, 0] = 1

# quantum kernel function

@qml.qnode(dev_kernel)
def kernel(x1, x2):
    qml.templates.MottonenStatePreparation(x1, wires=range(n_qubits))
    qml.adjoint(qml.templates.MottonenStatePreparation(x2, wires=range(n_qubits)))
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

#Compute the matrix whose entries are the kernel.
def kernel_matrix(A, B):
    
    return np.array([[kernel(a, b) for b in B] for a in A])

"""
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the observations (y1, X1), 
    and the prior kernel function.
"""
def GP(X1, y1, X2, noise):
    
    # Kernel of the observations(train)
    Σ11 = kernel_matrix(X1, X1) + (noise**2) * np.eye(len(X1)) 

    # Kernel of observations(train) vs to-predict(test)
    Σ12 = kernel_matrix(X1, X2) 

    #Kernel of observations(test)
    Σ22 = kernel_matrix(X2, X2)

    Σ11_inv = np.linalg.inv(Σ11)

    μ2 = np.transpose(Σ12) @ Σ11_inv @ y1

    Σ2 = Σ22 + (noise**2) * np.eye(len(X2))  - np.transpose(Σ12) @ Σ11_inv @ Σ12

    return μ2, Σ2 



def sigmoid(x):

    return 1 / (1 + math.exp(-x))


"""Compute the mean and covariance at the test points"""

mu, cov = GP(X_train, y_train_new, X_test, 0.4)

# compute variance

var = np.diag(cov)

#print(mu, var)


"""
Computes the probability of t=1 at points X_test
given training data X, t and kernel parameters theta.
"""

def ff(a_mu, a_var):
    
    
    kappa = 1.0 / np.sqrt(1.0 + np.pi * np.sqrt(np.abs(a_var)) / 8)

    return sigmoid(kappa * a_mu)

f2 = np.vectorize(ff)

prediction = f2(mu, var)

C = []


for i in prediction:
    if i>=0.5:
        C.append(1)
    else:
        C.append(0)

print(C)


BB = metrics.confusion_matrix(y_test, C, labels=[0,1])



print(BB) 