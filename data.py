import pandas as pd
import numpy as np
import utils
import math

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
print('training data ready with size:', train.shape)
test = data[m_train:,:]
print('test data ready with size:', test.shape)
