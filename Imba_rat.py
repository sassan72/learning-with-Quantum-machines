
"""This code determine the number of train 
class samples with label 0 and number of train 
class samples with label 1. The imbalance ration 
can be calculated easily."""  

import numpy as np

import sklearn.decomposition
from sklearn import metrics

import math


data1 = np.loadtxt("CERVICAL-F16-REC-ORIGINAL-1x.txt") # import pandas as pd

data2 = np.loadtxt("CERVICAL-F16-REC-ORIGINAL-1y.txt") #

data3 = np.loadtxt("CERVICAL-F16-REC-ORIGINAL-1x-test.txt") #

data4 = np.loadtxt("CERVICAL-F16-REC-ORIGINAL-1y-test.txt") #

y_train= data2[:,] 

X_train2=data1[:]

y_test= data4[:,] 

X_test2=data3[:]



pi = np.pi


data1 = np.loadtxt("TDS.txt") # import pandas as pd

data2 = np.loadtxt("TLD.txt") #

data3 = np.loadtxt("VDS.txt") #

data4 = np.loadtxt("VLD.txt") #

y_train_new1 = []

y_train_new2 = []

for i in range(len(X_train)):
    if y_train[i]==1:
        a= 1
        y_train_new1.append(a)
    else:
        b= 0
        y_train_new2.append(b)



print(len(y_train_new1), len(y_train_new2))