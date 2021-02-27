import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import math

np.random.seed(5)

print('loading the data ...')
data = pd.read_excel('./data.xlsx', header=1)
data = np.array(data)
data = data[:,1:]
m = data.shape[0]
print('data loaded with size: ', data.shape)

if(utils.checkIfThereIsNan(data)):
    print('there is Nan in data')
else:
    print('data is good')

print('shuffling the data ...')
np.random.shuffle(data)
print('data shuffled')

print('dividing the data ...')
m_train = math.ceil(m * 0.75)
train = data[0:m_train,:]
x_train = train[:,0:6]
y_train = train[:,6:8]
n_train = x_train.shape[1]
print('training data ready with size:', train.shape)
print('training input size:', x_train.shape)
print('training output size:', y_train.shape)
test = data[m_train:,:]
x_test = test[:,0:6]
y_test = test[:,6:8]
print('test data ready with size:', test.shape)
print('test input size:', x_test.shape)
print('test output size:', y_test.shape)

#print('plot shows cement before and after normalization')
#utils.plotSingleRow(x_train[:,0], "cement o.p.c")
#print('normalizing data ...')
#x_train = ((x_train - x_train.min())/(x_train.max() - x_train.min()))
#print('data normalized ')
#utils.plotSingleRow(x_train[:,0], "cement o.p.c")

print('initializing model ...')
learning_rate = 0.05
epochs = 2000
n1 = 10
n2 = 5
n3 = 2
w1 = np.random.rand(n_train, n1) # (6,10)
b1 = np.zeros((n1))              # (10,1)
w2 = np.random.rand(n1, n2)      # (10,5)
b2 = np.zeros((n2))              # (5,1)
w3 = np.random.rand(n2, n3)      # (5,2)
b3 = np.zeros((n3))              # (2,1)

print('started trainning ...')
for i in range(epochs):
    A3, Z3, A2, Z2, A1, Z1 = utils.forwadProbagationStep(x_train, w1, b1, w2, b2, w3, b3)
    dw3 , db3, dw2, db2, dw1, db1 = utils.backwardPropagationStep(y_train, A3, Z3, w3, A2, Z2, w2, A1, Z1, x_train)

    w3 = w3 - (learning_rate * dw3)
    b3 = b3 - (learning_rate * db3)
    w2 = w2 - (learning_rate * dw2)
    b2 = b2 - (learning_rate * db2)
    w1 = w1 - (learning_rate * dw1)
    b1 = b1 - (learning_rate * db1)

    cost =  utils.costFunction(y_train, A3)
    accuracy = utils.get_accuracy_value(y_train, A3)
    print('epoch', i, 'cost:', cost, 'accuracy:', accuracy)    
