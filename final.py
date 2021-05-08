import os
import data
from utils import utils
import numpy as np
import tensorflow as tf
from keras.models import Sequential

tf.compat.v1.disable_eager_execution()


NUM_EPOCHS = 10000
SAVE_INPUT = False
SAVE_OUTPUT = True
SAVE_WEIGHTS = True
LOAD_WEIGHTS = True
INPUT_TITLE = 'data_files/our_data3.xlsx'
OUTPUT_TITLE = 'results/our_data1.xlsx'
TRAIN_MODEL_1 = False
TRAIN_MODEL_2 = False
TRAIN_MODEL_3 = True


if (TRAIN_MODEL_1):
	train1, test1 = data.getFinalData(INPUT_TITLE)
	train1, test1 = data.prepareMultipleData(train1, test1, [16,17])
	pred1 = data.getTestData([16,17])
	print('\n train1:', train1.shape)
	xr1 = train1[:,0:14]
	yr1 = train1[:,14:16]
	xt1 = test1[:,0:14]
	yt1 = test1[:,14:16]
	xp1 = pred1[:,0:14]
	yp1 = pred1[:,14,16]
	utils.exceptionIfNan(train1)
	utils.exceptionIfNan(test1)
	utils.exceptionIfNan(pred1)
	print('data 1 ready with train:', train1.shape, 'and test:', test1.shape)
	if(SAVE_INPUT):
		data.saveData(train1, 'train1.xlsx')

if (TRAIN_MODEL_2):
	train2, test2 = data.getFinalData(INPUT_TITLE)
	train2, test2 = data.prepareMultipleData(train2, test2, [14, 15, 17])
	pred2 = data.getTestData([14, 15, 17])
	xr2 = train2[:,0:14]
	yr2 = train2[:,14:15]
	xt2 = test2[:,0:14]
	yt2 = test2[:,14:15]
	xp2 = pred2[:,0:14]
	yp2 = pred2[:,14:15]
	utils.exceptionIfNan(train2)
	utils.exceptionIfNan(test2)
	utils.exceptionIfNan(pred2)
	print('data 2 ready with train:', train2.shape, 'and test:', test2.shape)
	if(SAVE_INPUT):
		data.saveData(train2, 'train2.xlsx')

if (TRAIN_MODEL_3):
	train3, test3 = data.getFinalData(INPUT_TITLE)
	train3, test3 = data.prepareMultipleData(train3, test3, [14, 15, 16])
	pred3 = data.getTestData([14, 15, 16])
	xr3 = train3[:,0:14]
	yr3 = train3[:,14:15]
	xt3 = test3[:,0:14]
	yt3 = test3[:,14:15]
	xp3 = pred3[:,0:14]
	yp3 = pred3[:,14:15]
	utils.exceptionIfNan(train3)
	utils.exceptionIfNan(test3)
	utils.exceptionIfNan(pred3)
	print('data 3 ready with train:', train3.shape, 'and test:', test3.shape)
	if(SAVE_INPUT):
		data.saveData(train3, 'train3.xlsx')



model1 = utils.newSeqentialModel(14, 2)
model2 = utils.newSeqentialModel(14, 1)
model3 = utils.newSeqentialModel(14, 1)







if (TRAIN_MODEL_1):
	if(LOAD_WEIGHTS):
		print('loading model weights ...')
		output_dir = os.path.join(os.getcwd(), "saved_wights")
		model1.load_weights(filepath=os.path.join(output_dir, "all_data1_1.h5"))
		print('model weights loaded')

	print('\n\nTrainning model 1')
	model1.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
	model1.fit(xr1, yr1, epochs=NUM_EPOCHS, batch_size=32, validation_data=(xt1, yt1), verbose=1)

	_, accuracy_test_1 = model1.evaluate(xp1, yp1)
	print('model 1 trained')
	print('Accuracy on test data: %.2f' % (accuracy_test_1))

	test_predictions = np.around(model1.predict(xp1), 1)

	for i in range(20):
		print(' test: predicted:', test_predictions[i], 'real data:', yp1[i])

	if (SAVE_OUTPUT):
		result = np.concatenate([xp1, yp1, test_predictions], axis=1)
		data.saveData(result, OUTPUT_TITLE)

	if(SAVE_WEIGHTS):
		print('saving model weights ...')
		output_dir = os.path.join(os.getcwd(), "saved_wights")
		model1.save_weights(filepath=os.path.join(output_dir, "all_data1_1.h5"))
		print('model weights saved')



if (TRAIN_MODEL_2):
	if(LOAD_WEIGHTS):
		print('loading model weights ...')
		output_dir = os.path.join(os.getcwd(), "saved_wights")
		model2.load_weights(filepath=os.path.join(output_dir, "all_data1_2.h5"))
		print('model weights loaded')

	print('\n\nTrainning model 2')
	model2.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
	model2.fit(xr2, yr2, epochs=NUM_EPOCHS, batch_size=32, validation_data=(xt2, yt2), verbose=1)

	_, accuracy_test_2 = model2.evaluate(xp2, yp2)
	print('model 2 trained')
	print('Accuracy on test data: %.2f' % (accuracy_test_2))

	test_predictions = np.around(model2.predict(xp2), 1)

	for i in range(20):
		print(' test: predicted:', test_predictions[i], 'real data:', yp2[i])

	if (SAVE_OUTPUT):
		result = np.concatenate([xp2, yp2, test_predictions], axis=1)
		data.saveData(result, OUTPUT_TITLE)

	if(SAVE_WEIGHTS):
		print('saving model weights ...')
		output_dir = os.path.join(os.getcwd(), "saved_wights")
		model2.save_weights(filepath=os.path.join(output_dir, "all_data1_2.h5"))
		print('model weights saved')





if (TRAIN_MODEL_3):
	if(LOAD_WEIGHTS):
		print('loading model weights ...')
		output_dir = os.path.join(os.getcwd(), "saved_wights")
		model3.load_weights(filepath=os.path.join(output_dir, "all_data1_3.h5"))
		print('model weights loaded')

	print('\n\nTrainning model 3')
	model3.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
	model3.fit(xr3, yr3, epochs=NUM_EPOCHS, batch_size=64, validation_data=(xt3, yt3), verbose=1)

	_, accuracy_test_3 = model3.evaluate(xp3, yp3)
	print('model 3 trained')
	print('Accuracy on test data: %.2f' % (accuracy_test_3))

	test_predictions = np.around(model3.predict(xp3), 1)

	for i in range(20):
		print(' test: predicted:', test_predictions[i], 'real data:', yp3[i])

	if (SAVE_OUTPUT):
		result = np.concatenate([xp3, yp3, test_predictions], axis=1)
		data.saveData(result, OUTPUT_TITLE)

	if(SAVE_WEIGHTS):
		print('saving model weights ...')
		output_dir = os.path.join(os.getcwd(), "saved_wights")
		model3.save_weights(filepath=os.path.join(output_dir, "all_data1_3.h5"))
		print('model weights saved')
