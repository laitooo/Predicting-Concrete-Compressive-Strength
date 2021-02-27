import numpy as np
import matplotlib.pyplot as plt

def checkIfThereIsNan(array):
    for i in range(array.shape[0]):
        if(np.isnan(array[i]).any(axis=0)):
            return True
    return False

def plotSingleRow(array, ylabel):
    plt.figure()
    plt.plot(array)
    plt.ylabel(ylabel)
    plt.show()

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def backwardSigmoid(Z):
    return (sigmoid(Z)* (1 - sigmoid(Z)))

def forwadProbagationStep(x, w1, b1, w2, b2, w3, b3):
    Z1 = x.dot(w1) + b1
    A1 = sigmoid(Z1)
    Z2 = A1.dot(w2) + b2
    A2 = sigmoid(Z2)
    Z3 = A2.dot(w3) + b3
    A3 = sigmoid(Z3)
    #A3 = Z3
    return A3, Z3, A2, Z2, A1, Z1

def costFunction(y_real, y_generated):
    m = y_real.shape[0]
    cost = (-y_real.dot(np.log(y_generated.T)) - (1 - y_real).dot(np.log(1 - y_generated).T))
    cost = (1 / m) * np.sum(cost)
    return cost

def get_accuracy_value(y_real, y_generated):
    y_temp = y_real == y_generated
    return np.count_nonzero(y_temp)

def backwardPropagationStep(y_train, A3, Z3, w3, A2, Z2, w2, A1, Z1, x_train):
    m = y_train.shape[0]
    delta3 = (y_train - A3) * backwardSigmoid(A3) * Z3
    #delta3 = (y_train - A3)
    delta2 = np.multiply(np.multiply(delta3.dot(w3.T), backwardSigmoid(A2)), Z2)
    delta1 = np.multiply(np.multiply(delta2.dot(w2.T), backwardSigmoid(A1)), Z1)

    dw3 = A2.T.dot(delta3)/m
    db3 = (np.ones((m, 1))/m).T.dot(delta3)
    dw2 = A1.T.dot(delta2)/m
    db2 = (np.ones((m, 1))/m).T.dot(delta2)
    dw1 = x_train.T.dot(delta1)/m
    db1 = (np.ones((m, 1))/m).T.dot(delta1)
    return dw3, db3, dw2, db2, dw1, db1
