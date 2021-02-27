import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import utils
import data

x_train, y_train, x_test, y_test, n = data.getData()

#print('plot shows cement before and after normalization')
#utils.plotSingleRow(x_train[:,0], "cement o.p.c")
#print('normalizing data ...')
#x_train = ((x_train - x_train.min())/(x_train.max() - x_train.min()))
#print('data normalized ')
#utils.plotSingleRow(x_train[:,0], "cement o.p.c")

print('initializing model ...')
learning_rate = 0.05
epochs = 200
n1 = 10
n2 = 5
n3 = 2
w1 = np.random.rand(n, n1) # (6,10)
b1 = np.zeros((n1))              # (10,1)
w2 = np.random.rand(n1, n2)      # (10,5)
b2 = np.zeros((n2))              # (5,1)
w3 = np.random.rand(n2, n3)      # (5,2)
b3 = np.zeros((n3))              # (2,1)

print('started trainning ...')
for i in range(epochs+1):
    A3, Z3, A2, Z2, A1, Z1 = utils.forwadProbagationStep(x_train, w1, b1, w2, b2, w3, b3)

    cost =  utils.costFunction(y_train, A3)
    accuracy = utils.get_accuracy_value(y_train, A3)
    if(i%100 == 0):
        print('epoch', i, 'cost:', cost, 'accuracy:', accuracy)    

    dw3 , db3, dw2, db2, dw1, db1 = utils.backwardPropagationStep(y_train, A3, Z3, w3, A2, Z2, w2, A1, Z1, x_train)

    w3 = w3 + (learning_rate * dw3)
    b3 = b3 + (learning_rate * db3)
    w2 = w2 + (learning_rate * dw2)
    b2 = b2 + (learning_rate * db2)
    w1 = w1 + (learning_rate * dw1)
    b1 = b1 + (learning_rate * db1)
    #print('np.sum(dw3*alpha):', learning_rate*np.sum(dw3))

print('testing againsta random test sample ...')
idx = np.random.randint(x_test.shape[0])
x_tmp = x_test[idx,:]
y_tmp = y_test[idx,:]
result, _, _, _, _, _ = utils.forwadProbagationStep(x_tmp, w1, b1, w2, b2, w3, b3)
print('real outpt:', y_tmp)
print('generated output', result)
