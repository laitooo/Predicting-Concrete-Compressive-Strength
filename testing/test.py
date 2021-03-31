import pandas as pd
import numpy as np
import math
import tensorflow as tf
from ..utils import utils
from keras.models import Sequential
from keras.layers import Dense, LayerNormalization, Dropout

num_epochs = 2000

print('loading the data ...')
data = pd.read_excel('testing/test_data_v1.xlsx', header=1)   
data = np.array(data)
#data = data[:,1:]
m = data.shape[0]
print('data loaded with size: ', data.shape)

if(checkIfThereIsNan(data)):
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
x_train = train[:,0:12]
y_train = train[:,12:13]
print('training data ready with size:', train.shape)
print('training input size:', x_train.shape)
print('training output size:', y_train.shape)
test = data[m_train:,:]
x_test = test[:,0:12]
y_test = test[:,12:13]
n = x_train.shape[1]
print('test data ready with size:', test.shape)
print('test input size:', x_test.shape)
print('test output size:', y_test.shape)


print('loading the data2 ...')
data2 = pd.read_excel('./test_data_v2.xlsx', header=1)   
data2 = np.array(data2)
#data = data[:,1:]
m2 = data2.shape[0]
print('data2 loaded with size: ', data2.shape)

if(utils.checkIfThereIsNan(data2)):
    print('there is Nan in data2')
else:
    print('data2 is good')

np.random.shuffle(data2)
print('shuffling the data2 ...')
np.random.shuffle(data2)
print('data2 shuffled')

print('dividing the data2 ...')
m_train2 = math.ceil(m2 * 0.75)
train2 = data2[0:m_train2,:]
x_train2 = train2[:,0:10]
y_train2 = train2[:,10:11]
print('training data2 ready with size:', train2.shape)
print('training input size:', x_train2.shape)
print('training output size:', y_train2.shape)
test2 = data2[m_train2:,:]
x_test2 = test2[:,0:10]
y_test2 = test2[:,10:11]
n2 = x_train2.shape[1]
print('test data ready with size:', test.shape)
print('test input size:', x_test.shape)
print('test output size:', y_test.shape)


tf.compat.v1.disable_eager_execution()

model = Sequential()
model.add(LayerNormalization(input_dim=12))
model.add(Dense(160))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model2 = Sequential()
model2.add(LayerNormalization(input_dim=10))
model2.add(Dense(160))
model2.add(Dense(80, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(40, activation='sigmoid'))
model2.add(Dense(20, activation='relu'))
model2.add(Dense(10, activation='sigmoid'))
model2.add(Dense(5, activation='relu'))
model2.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
model.fit(x_train, y_train, epochs=num_epochs, batch_size=32, validation_data=(x_test, y_test))
print('starting training ...')

_, accuracy_train = model.evaluate(x_train, y_train)
_, accuracy_test = model.evaluate(x_test, y_test)
print('model trained')

model2.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
model2.fit(x_train2, y_train2, epochs=num_epochs, batch_size=32, validation_data=(x_test2, y_test2))
print('starting training ...')


_, accuracy_train2 = model2.evaluate(x_train2, y_train2)
_, accuracy_test2 = model2.evaluate(x_test2, y_test2)
print('model2 trained')


print('Accuracy on train data: %.2f' % (accuracy_train*100))
print('Accuracy on test data: %.2f' % (accuracy_test*100))
print('Accuracy on train data2: %.2f' % (accuracy_train2*100))
print('Accuracy on test data2: %.2f' % (accuracy_test2*100))

def convert(A):
    return np.delete(np.delete(A, 2, 1), 2, 1)

train_predictions = np.around(model.predict(x_train), 1)
test_predictions = np.around(model.predict(x_test), 1)
train_predictions2 = np.around(model2.predict(convert(x_train)), 1)
test_predictions2 = np.around(model2.predict(convert(x_test)), 1)

for i in range(20):
	print('train :', train_predictions[i], 'train2 :', train_predictions2[i],
     ' real data :', y_train[i] , ' test :', test_predictions[i], ' test2 :', test_predictions2[i],
      'real data :', y_test[i])


df = pd.DataFrame(np.concatenate([train_predictions, train_predictions2, y_train], axis=1))
filepath = 'testing_train_outout.xlsx'
df.to_excel(filepath, index=False)
print('saved testing train to excel file')

df = pd.DataFrame(np.concatenate([test_predictions, test_predictions2, y_test], axis=1))
filepath = 'testing_test_outout.xlsx'
df.to_excel(filepath, index=False)
print('saved testing test to excel file')