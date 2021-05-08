import pandas as pd
import numpy as np
import math
from utils import utils

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

def getFinalData(input):
    print('loading the data ...')
    data = pd.read_excel(input, header=1)
    data = np.array(data)
    m = data.shape[0]
    print('data loaded with size: ', data.shape, '\n\n')

    if(utils.checkIfThereIsNan(data)):
        print('there is Nan in data')
    else:
        print('data is good')

    np.random.shuffle(data)
    print('\n\nshuffling the data ...')
    np.random.shuffle(data)
    print('data shuffled')

    print('\n\ndividing the data ...')
    m_train = math.ceil(m * 0.75)
    train = data[0:m_train,:]
    test = data[m_train:,:]
    print('training data ready with size:', train.shape)
    print('test data ready with size:', test.shape)
    return train, test


def prepareData(test, removedColumns):
    test = np.delete(test, removedColumns, axis=1)
    test = np.array(pd.DataFrame(test).dropna(axis=0))
    return test

def prepareMultipleData(train, test, removedColumns):
    train = np.delete(train, removedColumns, axis=1)
    test = np.delete(test, removedColumns, axis=1)
    train = np.array(pd.DataFrame(train).dropna(axis=0))
    test = np.array(pd.DataFrame(test).dropna(axis=0))
    return train, test

def saveData(data, path):
    df = pd.DataFrame(data)
    df.to_excel(path, index=False)
    print('saved data to excel file \npath:', path)

def getRandomTestData(input):
    print('loading the data ...')
    data = pd.read_excel(input, header=1)
    data = np.array(data)
    m = data.shape[0]
    print('all data size: ', data.shape, '\n\n')

    if(utils.checkIfThereIsNan(data)):
        print('there is Nan in data')
    else:
        print('data is good')

    np.random.shuffle(data)
    print('\n\nshuffed the data ...')

    print('\n\ndividing the data ...')
    m_test = math.ceil(m * 0.15)
    test = data[0:m_test,:]
    print('test data size:', test.shape)
    return test

def getTestData(removedColumns):
    data = pd.read_excel('data_files/test_data.xlsx', header=0)
    data = np.array(data)
    data = prepareData(data, removedColumns)
    return data
    
