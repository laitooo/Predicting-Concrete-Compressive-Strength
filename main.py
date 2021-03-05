import data
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LayerNormalization

x_train, y_train, x_test, y_test, _ = data.getData()

print('preparing model ...')

model = Sequential()
model.add(LayerNormalization(input_dim=6))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(2))


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5000, batch_size=32, validation_data=(x_test, y_test))
print('starting training ...')

_, accuracy_train = model.evaluate(x_train, y_train)
_, accuracy_test = model.evaluate(x_test, y_test)
print('model trained')

print('Accuracy on train data: %.2f' % (accuracy_train*100))
print('Accuracy on test data: %.2f' % (accuracy_test*100))

predictions = model.predict(x_train)

for i in range(20):
	print('x:', x_train[i,:], ' predicted:', predictions[i], ' y:', y_train[i] )