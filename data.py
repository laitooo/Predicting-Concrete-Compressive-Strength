import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import math

np.random.seed(3)

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
print(x_train[:,0].shape)
#utils.plotSingleRow(x_train[:,0], "cement o.p.c")
print('normalizing data ...')
x_train = ((x_train - x_train.min())/(x_train.max() - x_train.min()))
print('data normalized ')
#utils.plotSingleRow(x_train[:,0], "cement o.p.c")
