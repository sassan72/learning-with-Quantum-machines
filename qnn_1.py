
#Hybrid quantum varitional classifier

import pennylane as qml
import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import sklearn.decomposition
from sklearn import metrics


np.random.seed(0)
torch.manual_seed(0)

num_classes = 2
margin = 0.15
feature_size = 8
batch_size = 100
lr_adam = 0.001
# the number of the required qubits is calculated from the number of features
num_qubits = int(np.ceil(np.log2(feature_size)))
num_layers = 4
total_iterations = 200

dev = qml.device("default.qubit", wires=num_qubits)



#################################################################################
# Quantum Circuit
# We first create the layer that will be repeated in our variational quantum
# circuits. It consists of rotation gates for each qubit, followed by
# entangling/CNOT gates


def layer(W):
    for i in range(num_qubits):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
    for j in range(num_qubits - 1):
        qml.CNOT(wires=[j, j + 1])
    if num_qubits >= 2:
        # Apply additional CNOT to entangle the last with the first qubit
        qml.CNOT(wires=[num_qubits - 1, 0])


#################################################################################
#the whole quantum circuit for data encoding and parametrized parts    


def circuit(weights, feat=None):
    qml.templates.MottonenStatePreparation(feat, range(num_qubits))
    for W in weights:
        layer(W)

    return qml.expval(qml.PauliZ(0))


qnodes = []
for iq in range(num_classes):
    qnode = qml.QNode(circuit, dev, interface="torch")
    qnodes.append(qnode)


#################################################################################
# The variational quantum circuit is parametrized by the weights. We use a
# classical bias term that is applied after processing the quantum circuit's
# output. Both variational circuit weights and classical bias term are optimized.


def variational_classifier(q_circuit, params, feat):
    weights = params[0]
    bias = params[1]
    return q_circuit(weights, feat=feat) + bias


##############################################################################
# Loss Function

def class_loss(q_circuits, all_params, feature_vecs, true_labels):
    loss = 0
    num_samples = len(true_labels)
    for i, feature_vec in enumerate(feature_vecs):
        # Compute the score given to this sample by the classifier corresponding to the
        # true label. So for a true label of 1, get the score computed by classifer 1,
        # which distinguishes between "class 1" or "not class 1".
        s_true = variational_classifier(
            q_circuits[int(true_labels[i])],
            (all_params[0][int(true_labels[i])], all_params[1][int(true_labels[i])]),
            feature_vec,
        )
        s_true = s_true.float()
        li = 0

        for j in range(num_classes):
            if j != int(true_labels[i]):
                s_j = variational_classifier(
                    q_circuits[j], (all_params[0][j], all_params[1][j]), feature_vec
                )
                s_j = s_j.float()
                li += torch.max(torch.zeros(1).float(), s_j - s_true + margin)
        loss += li

    return loss / num_samples


##########################################################################################
# Classification Function
# Next, we use the learned models to classify our samples. 

def classify(q_circuits, all_params, feature_vecs, labels):
    predicted_labels = []
    for i, feature_vec in enumerate(feature_vecs):
        scores = np.zeros(num_classes)
        for c in range(num_classes):
            score = variational_classifier(
                q_circuits[c], (all_params[0][c], all_params[1][c]), feature_vec
            )
            scores[c] = float(score)
        pred_class = np.argmax(scores)
        predicted_labels.append(pred_class)
    return predicted_labels


def accuracy(labels, hard_predictions):
    loss = 0
    for l, p in zip(labels, hard_predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / labels.shape[0]
    return loss


#################################################################################
# Data Loading and Processing. Run data as txt file.

data1 = np.loadtxt("TDS.txt") # training data
data2 = np.loadtxt("TLD.txt") # training label

data3 = np.loadtxt("VDS.txt") # validation data

data4 = np.loadtxt("VLD.txt") # validation label

y_train= data2[:,] 

X_train2=data1[:]

y_test= data4[:,] 

X_test2=data3[:]

#Data normalization

X_train1 = sklearn.preprocessing.normalize(X_train2, norm='l2',axis=0) 

X_test1 = sklearn.preprocessing.normalize(X_test2, norm='l2', axis=0)

X_train = sklearn.preprocessing.normalize(X_train1, norm='l2',axis=1)

X_test = sklearn.preprocessing.normalize(X_test1, norm='l2', axis=1)


#################################################################################
# Training Procedure
# In the training procedure, we begin by first initializing randomly the parameters
# we want to learn (variational circuit weights and classical bias). 


def training():
    num_train = y_train.shape[0]

    q_circuits = qnodes

    # Initialize the parameters
    all_weights = [
        Variable(0.1 * torch.randn(num_layers, num_qubits, 3), requires_grad=True)
        for i in range(num_classes)
    ]
    all_bias = [Variable(0.1 * torch.ones(1), requires_grad=True) for i in range(num_classes)]
    optimizer = optim.Adam(all_weights + all_bias, lr=lr_adam)
    params = (all_weights, all_bias)
    

    costs, train_acc, test_acc = [], [], []

    # train the variational classifier
    for it in range(total_iterations):
        batch_index = np.random.randint(0, num_train, (batch_size,))
        X_train_batch = X_train[batch_index]
        Y_train_batch = y_train[batch_index]

        optimizer.zero_grad()
        curr_cost = class_loss(q_circuits, params, X_train_batch, Y_train_batch)
        curr_cost.backward()
        optimizer.step()

        # Compute predictions on train and validation set
        predictions_train = classify(q_circuits, params, X_train, y_train)
        predictions_test = classify(q_circuits, params, X_test, y_test)
        acc_train = accuracy(y_train, predictions_train)
        acc_test = accuracy(y_test, predictions_test)
        #BB = metrics.confusion_matrix(y_test, predictions_test, labels=[0,1])

        print(
            "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc test: {:0.7f} | Prediction_test: {} " 
            "".format(it + 1, curr_cost.item(), acc_train, acc_test, predictions_test)
        )

        costs.append(curr_cost.item())
        train_acc.append(acc_train)
        test_acc.append(acc_test)


    return costs, train_acc, test_acc, predictions_test

print(training())