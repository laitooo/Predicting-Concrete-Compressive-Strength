import data
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

x_train, y_train, x_test, y_test, _ = data.getTestData()

print('preparing model ...')

model = Sequential()
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(5, activation='relu'))
model.add(Dense(2))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2000, batch_size=64)
print('starting training ...')

_, accuracy_train = model.evaluate(x_train, y_train)
_, accuracy_test = model.evaluate(x_test, y_test)
print('model trained')

print('Accuracy on train data: %.2f' % (accuracy_train*100))
print('Accuracy on test data: %.2f' % (accuracy_test*100))