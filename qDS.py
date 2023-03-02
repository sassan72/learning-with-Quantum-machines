import numpy as np
import pennylane as qml
import matplotlib
import sklearn.datasets
import sklearn.decomposition

import math
from sklearn import metrics



num_qubits = 4


dev = qml.device('default.mixed', wires=num_qubits, shots=8192)

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

x1_train=[]
x2_train=[]

for i in range(len(X_train)):
    if y_train[i]==1:
        a=X_train[i]
        x1_train.append(a)
    else:
        b=X_train[i]
        x2_train.append(b)
        
def ops(X):
    
    qml.templates.MottonenStatePreparation(X, wires=[1, 2, 3])
    

ops1 = qml.ctrl(ops, control=0)


@qml.qnode(dev)
def circuit(X1, X2):
    qml.Hadamard(wires=0)
    ops1(X1)
    
    qml.PauliX(wires=0)
    
    ops1(X2)
    
    
    qml.PauliX(wires=0)


    qml.Hadamard(wires=0)
    
    
    return qml.expval(qml.PauliZ(0))


D1 = []

for x1 in x1_train:
    for x2 in X_test:
        a = circuit(x1, x2)
        D1.append(a)


D2 = []

for x1 in x2_train:
    for x2 in X_test:
        b = circuit(x1, x2)
        D2.append(b)


D11 = np.reshape(D1, (len(x1_train), len(X_test)))
D22 = np.reshape(D2, (len(x2_train), len(X_test)))

d11 = np.max(D11, axis=0)
d22 = np.max(D22, axis=0)
m = len(X_test)

Prediction=[]
for i in range(m):
    if d11[i] >= d22[i]:
        Prediction.append(1)
    else:
        Prediction.append(0)
    
BB = metrics.confusion_matrix(y_test, Prediction, labels=[0,1])

print(Prediction)



print(BB)      