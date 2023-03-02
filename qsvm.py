import numpy as np

from sklearn.svm import SVC

import pennylane as qml
from pennylane.operation import Tensor

import matplotlib.pyplot as plt
import sklearn.decomposition
from sklearn import metrics


# data preprocessing

data1 = np.loadtxt("TDS.txt") # import pandas as pd

data2 = np.loadtxt("TLD.txt") #

data3 = np.loadtxt("VDS.txt") #

data4 = np.loadtxt("VLD.txt") #

y_train= data2[:,] 

X_train2=data1[:]

y_test= data4[:,] 

X_test2=data3[:]


# data normalization

X_train1 = sklearn.preprocessing.normalize(X_train2, norm='l2',axis=0)

X_test1 = sklearn.preprocessing.normalize(X_test2, norm='l2', axis=0)

X_train = sklearn.preprocessing.normalize(X_train1, norm='l2',axis=1)

X_test = sklearn.preprocessing.normalize(X_test1, norm='l2', axis=1)


n_qubits = 3

pi = np.pi


# Define the quantum simulator environemnt.

dev_kernel = qml.device("default.qubit", wires=n_qubits)


projector = np.zeros((2**n_qubits, 2**n_qubits))
projector[0, 0] = 1

"""Build the quantum kernel function."""

@qml.qnode(dev_kernel)
def kernel(x1, x2):
    
    qml.templates.MottonenStatePreparation(x1, wires=range(n_qubits))
    qml.adjoint(qml.templates.MottonenStatePreparation(x2, wires=range(n_qubits)))
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

"""Compute the matrix whose entries are the kernel
evaluated on pairwise data."""

def kernel_matrix(A, B):
    
    return np.array([[kernel(a, b) for b in B] for a in A])
######################################################################
# Training the SVM optimizes internal parameters that basically 
# weigh kernel functions. 


svm = SVC(kernel=kernel_matrix, class_weight='balanced').fit(X_train, y_train)

######################################################################
# Let’s compute the accuracy on the test set.

predictions = svm.predict(X_test)
print(predictions)

BB = metrics.confusion_matrix(y_test, predictions, labels=[0,1])

print(BB)