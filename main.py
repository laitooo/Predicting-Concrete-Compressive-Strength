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
print('normalizing data ...')
x_train = ((x_train - x_train.min())/(x_train.max() - x_train.min()))
print('data normalized ')
#utils.plotSingleRow(x_train[:,0], "cement o.p.c")

print('initializing model ...')
n1 = 10
n2 = 5
n3 = 2
w1 = np.random.rand(n_train, n1) # (6,10)
b1 = np.random.rand(n1)          # (10,1)
w2 = np.random.rand(n1, n2)      # (10,5)
b2 = np.random.rand(n2)          # (5,1)
w3 = np.random.rand(n2, n3)      # (5,2)
b3 = np.random.rand(n3)          # (2,1)

print('one backward propagation ...')
A3, Z3, A2, Z2, A1, Z1 = utils.forwadProbagationStep(x_train, w1, b1, w2, b2, w3, b3)
cost =  utils.costFunction(y_train, A3)
print('done.')
print('cost:', cost)

print('one backward propagation ...')
dw3 , db3, dw2, db2, dw1, db1 = utils.backwardPropagationStep(y_train, A3, Z3, w3, A2, Z2, w2, A1, Z1, x_train)
print('done.')