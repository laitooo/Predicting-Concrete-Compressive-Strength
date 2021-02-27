import pandas as pd
import numpy as np
import math
import utils

def getData():
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

    np.random.shuffle(data)
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
    n = x_train.shape[1]
    print('test data ready with size:', test.shape)
    print('test input size:', x_test.shape)
    print('test output size:', y_test.shape)
    return x_train, y_train, x_test, y_test, n

