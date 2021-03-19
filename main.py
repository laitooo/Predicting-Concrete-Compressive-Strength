import data
import utils
import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LayerNormalization, Dropout

SAVE_GRAPH = False
SAVE_WEIGHTS = False
SAVE_MODEL = False
LOAD_WEIGHTS = False

tf.compat.v1.disable_eager_execution()

x_train, y_train, x_test, y_test, _ = data.getData()

print('preparing model ...')

model = Sequential()
model.add(LayerNormalization(input_dim=6))
model.add(Dense(160))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(5, activation='relu'))
model.add(Dense(2))


if(LOAD_WEIGHTS):
	print('loading model weights ...')
	output_dir = os.path.join(os.getcwd(), "concrete")
	model.load_weights(filepath=os.path.join(output_dir, "wights.h5"))
	print('model weights loaded')

model.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_data=(x_test, y_test))
print('starting training ...')

if(SAVE_MODEL):
	print('saving model ...')
	model.save("my_model")
	print('model saved')

if(SAVE_WEIGHTS):
	print('saving model weights ...')
	output_dir = os.path.join(os.getcwd(), "concrete")
	model.save_weights(filepath=os.path.join(output_dir, "wights.h5"))
	print('model weights saved')

_, accuracy_train = model.evaluate(x_train, y_train)
_, accuracy_test = model.evaluate(x_test, y_test)
print('model trained')

print('Accuracy on train data: %.2f' % (accuracy_train*100))
print('Accuracy on test data: %.2f' % (accuracy_test*100))

train_predictions = np.around(model.predict(x_train), 1)
test_predictions = np.around(model.predict(x_test), 1)

for i in range(20):
	print('train : predicted:', train_predictions[i], ' real data:', y_train[i] , ' test: predicted:', 
		test_predictions[i], 'real data:', y_test[i])

if(SAVE_GRAPH):
	print('converting keras model to tensorflow ...')
	output_dir = os.path.join(os.getcwd(),"concrete")
	utils.keras_to_tensorflow(model,output_dir=output_dir,model_name="concrete.pb",log_tensorboard=True)
	print("MODEL SAVED")