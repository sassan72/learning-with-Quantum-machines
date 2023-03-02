import numpy as np
import pennylane as qml
import sklearn.decomposition
from sklearn import metrics
import math

"""This code is a simplified version of 
quantum kernel SVM, because it does not depend on optimization as thus SVM. 
It considers the imbalance ratio as hyperplane parameters."""


data1 = np.loadtxt("TDS.txt") # import pandas as pd

data2 = np.loadtxt("TLD.txt") #

data3 = np.loadtxt("VDS.txt") #

data4 = np.loadtxt("VLD.txt") # 

y_train= data2[:,] 

X_train2=data1[:]

y_test= data4[:,] 

X_test2=data3[:]


X_train1 = sklearn.preprocessing.normalize(X_train2, norm='l2',axis=0)

X_test1 = sklearn.preprocessing.normalize(X_test2, norm='l2', axis=0)

X_train = sklearn.preprocessing.normalize(X_train1, norm='l2',axis=1)

X_test = sklearn.preprocessing.normalize(X_test1, norm='l2', axis=1)


y_train_new =[]


for i in range(len(X_train)):
    if y_train[i]==1:
        a=0.5    # (1-imbalance_ratio)*y_train[i]
        y_train_new.append(a)
    else:
        b=-0.5   # imbalance_ratio*(-1)
        y_train_new.append(b)

feature_size = 8 # len(X_train[0])

n_qubits = 3 # log(feature_size)

pi = np.pi


dev = qml.device("default.qubit", wires=n_qubits, shots=8192) # for simulator


projector = np.zeros((2**n_qubits, 2**n_qubits))

projector[0, 0] = 1

"""Build the quantum kernel function."""
@qml.qnode(dev)
def kernel(x1, x2):
    
    qml.templates.MottonenStatePreparation(x1, wires=range(n_qubits))
    qml.adjoint(qml.templates.MottonenStatePreparation(x2, wires=range(n_qubits)))
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))


"""Compute the matrix whose entries are the kernel
evaluated on pairwise data from sets A and B."""

def kernel_matrix(A, B):
    
    return np.array([[kernel(a, b) for b in B] for a in A])


def GP(X1, y1, X2):
   
    Σ12 = kernel_matrix(X1, X2) #B
    

    μ =  y1 @ Σ12 

    

    return μ


μ = GP(X_train, y_train_new, X_test)


C = [] #prediction

for i in μ:

    if i>= 0:
        C.append(1)

    else:
        C.append(0)


print(C)

BB = metrics.confusion_matrix(y_test, C, labels=[0,1])


print(BB)

