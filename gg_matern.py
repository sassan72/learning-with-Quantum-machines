# this code calculate GD_q for classical Matern kernel.

import numpy as np

from sklearn.svm import SVC
from numpy.linalg import matrix_rank
import pennylane as qml
from pennylane.operation import Tensor

import sklearn.decomposition
from sklearn import metrics
from sklearn.metrics import hinge_loss
from scipy.linalg import sqrtm
from scipy.spatial.distance import pdist, cdist, squareform

from scipy.special import kv, gamma



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

dev_kernel = qml.device("default.qubit", wires=n_qubits)


projector = np.zeros((2**n_qubits, 2**n_qubits))
projector[0, 0] = 1

@qml.qnode(dev_kernel)
def kernel(x1, x2):
    """The quantum kernel."""
    qml.templates.MottonenStatePreparation(x1, wires=range(n_qubits))
    qml.adjoint(qml.templates.MottonenStatePreparation(x2, wires=range(n_qubits)))
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))


def kernel_matrix(A, B):
    """Compute the matrix whose entries are the kernel
       evaluated on pairwise data from sets A and B."""
    return np.array([[kernel(a, b) for b in B] for a in A])


K2 = kernel_matrix(X_train, X_train)





def matern(X, Y):

    dists = sklearn.metrics.pairwise_distances(X, Y, metric='euclidean')

    nu = 3.0

    K = dists
    K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
    tmp = np.sqrt(2 * nu) * K
    K.fill((2 ** (1.0 - nu)) / gamma(nu))
    K *= tmp**nu
    K *= kv(nu, tmp)


    return K


K1 = matern(X_train2, X_train2)


K22 = sqrtm(K2)

K11 = np.linalg.pinv(K1)

KK = (np.real(K22))@ K11 @ (np.real(K22)) # g(K^C||K^Q)

from scipy.linalg.interpolative import estimate_spectral_norm

NORM = estimate_spectral_norm(KK, its=100)

print(np.sqrt(NORM))
