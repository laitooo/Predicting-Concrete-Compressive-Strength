import math
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense, LayerNormalization, Dropout

tf.compat.v1.disable_eager_execution()

NUM_EPOCHS = 2000
TITILE = 'final_data3'
COMPLETE_USING_MEAN = False
COMPLETE_USING_MODE = False
COMPLETE_USING_REGRESSION = False

def checkIfThereIsNan(array):
    tmp = False
    for i in range(array.shape[1]):
        if (pd.isna(array[:,i]).any(axis=0)):
            print('\nn/a data in column', i)
            tmp = True
    return tmp

data = pd.read_excel('missing_data/' + TITILE + '.xlsx', header=0)
data = np.array(data)
x = data[:,0:21]
y = data[:,21:25]


x = x[:,4:21] # removed (index, date, location, type of coarse aggregate)
x = np.delete(x, 1, 1)  #removed type of fine aggregate
x = np.delete(x, 9, 1)  #removed additive name
x = np.delete(x, 9, 1)  #removed additive group letter
#x = np.delete(x, 8, 1)  #removed string : type of additive
#x = np.delete(x, 8, 1)  #removed string : dose of additive
#x = np.delete(x, 7, 1)  #removed string : water content
#x = np.delete(x, 11, 1) #removed int : hardened density

print('\nX[0]:', x[0,:], '\nY[0]:', y[0,:])

if(checkIfThereIsNan(x)):
    print('\nsome input data is missing')
else:
    print('\nall input data is good')


if(checkIfThereIsNan(y)):
    print('\nsome output data is missing')
else:
    print('\nall output data is good')

print('\n')
print('final data shape:', 'input:', x.shape, 'output:', y.shape)




if COMPLETE_USING_MEAN:
    print('\nCompleting data using the average')
    x_2 = np.zeros(x.shape)

    for i in range(x.shape[1]):
        tmp = x[:,i]
        mean = np.mean(tmp[~pd.isna(tmp)])
        correct = np.where(pd.isna(tmp), mean, tmp)
        x_2[:,i] = correct

    df = pd.DataFrame(x_2)
    filepath = 'missing_data/completed_' + TITILE + '_mean.xlsx'
    df.to_excel(filepath, index=False)
    print('saved completed data to excel file (using mean method)')



if COMPLETE_USING_MODE:
    print('\nCompleting data using the most appearing value')
    x_2 = np.zeros(x.shape)

    for i in range(x.shape[1]):
        tmp = x[:,i]
        mode = stats.mode(tmp[~pd.isna(tmp)])[0]
        correct = np.where(pd.isna(tmp), mode, tmp)
        x_2[:,i] = correct

    df = pd.DataFrame(x_2)
    filepath = 'missing_data/completed_' + TITILE + '_mode.xlsx'
    df.to_excel(filepath, index=False)
    print('saved completed data to excel file (using the most appearing value method)')






if COMPLETE_USING_REGRESSION:
    print('\nCompleting data using linear regression model')
    print('befor: x:', x.shape)
    tmp = pd.DataFrame(x).dropna(subset=[0,1,4,5,6,7,8,9,10])
    x_2 = np.array(tmp)
    x_2_full = np.empty((0,x_2.shape[1]))
    x_2_missing = np.empty((0,x_2.shape[1]))
    for i in range(x_2.shape[0]):
        if(pd.isna(x_2[i,:]).any()):
            x_2_missing = np.append(x_2_missing, [x_2[i,:]], axis=0)
        else:
            x_2_full = np.append(x_2_full, [x_2[i,:]], axis=0)

    print('after: x_2:', x_2.shape, '\nx_2_full', x_2_full.shape, '\nx_2_missing', x_2_missing.shape)

    x_3 = np.copy(x_2_full)
    x_3 = np.delete(x_3, 2, 1)
    x_3 = np.delete(x_3, 2, 1)
    y_3 = np.copy(x_2_full[:,2:4])

    x_4 = np.copy(x_2_missing)
    x_4 = np.delete(x_4, 2, 1)
    x_4 = np.delete(x_4, 2, 1)
    y_4 = np.copy(x_2_missing[:,2:4])

    model = Sequential()
    model.add(LayerNormalization(input_dim=9))
    model.add(Dense(80, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(40, activation='sigmoid'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(2))

    print('\n\nstarting training ...')
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
    model.fit(x_3, y_3, epochs=NUM_EPOCHS, batch_size=32, verbose=0)
    print('trainning finished')

    y_5 = np.zeros(y_4.shape)
    predictions = np.around(model.predict(x_4),1)



    for i in range(x_4.shape[0]):
        if(pd.isna(y_4[i,0])):
            y_5[i,0] = predictions[i,0]
        else:
            y_5[i,0] = y_4[i,0]
        if(pd.isna(y_4[i,1])):
            y_5[i,1] = predictions[i,1]
        else:
            y_5[i,1] = y_4[i,1]

    complete = np.concatenate([x_3[:,0:2], y_3, x_3[:,2:11]], axis=1)
    missing = np.concatenate([x_4[:,0:2], y_5, x_4[:,2:11]], axis=1)
    full_data = np.concatenate([complete, missing], axis= 0)

    df = pd.DataFrame(full_data)
    filepath = 'missing_data/completed_' + TITILE + '_regression.xlsx'
    df.to_excel(filepath, index=False)
    print('saved completed data to excel file (using linear regression method)')