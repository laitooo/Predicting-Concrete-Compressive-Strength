import os
import data
from utils import utils
import numpy as np
import tensorflow as tf
from keras.models import Sequential

tf.compat.v1.disable_eager_execution()


NUM_EPOCHS = 1000
SAVE_INPUT = False
SAVE_OUTPUT = True
SAVE_WEIGHTS = False
LOAD_WEIGHTS = True
INPUT_TITLE = 'data_files/amj_data3.xlsx'
OUTPUT_TITLE = 'results/amj_data1.xlsx'
TRAIN_MODEL_1 = False
TRAIN_MODEL_2 = False
TRAIN_MODEL_3 = True
PRINT_PRDECTIONS = True


if (TRAIN_MODEL_1):
	train1, test1, validation1 = data.getFinalData(INPUT_TITLE)
	train1, test1, validation1 = data.prepareMultipleData(train1, test1, validation1, [16,17])
	print('\n train1:', train1.shape)
	xr1 = train1[:,0:14]
	yr1 = train1[:,14:16]
	xt1 = test1[:,0:14]
	yt1 = test1[:,14,16]
	xv1 = validation1[:,0:14]
	yv1 = validation1[:,14:16]
	utils.exceptionIfNan(train1)
	utils.exceptionIfNan(test1)
	utils.exceptionIfNan(validation1)
	print('data 1 ready with train:', train1.shape, 'and test:', test1.shape, 'and validation:', 
	validation1.shape)
	if(SAVE_INPUT):
		data.saveData(train1, 'train1.xlsx')

if (TRAIN_MODEL_2):
	train2, test2, validation2 = data.getFinalData(INPUT_TITLE)
	train2, test2, validation2 = data.prepareMultipleData(train2, test2, validation2, [14, 15, 17])
	xr2 = train2[:,0:14]
	yr2 = train2[:,14:15]
	xt2 = test2[:,0:14]
	yt2 = test2[:,14:15]
	xv2 = validation2[:,0:14]
	yv2 = validation2[:,14:15]
	utils.exceptionIfNan(train2)
	utils.exceptionIfNan(test2)
	utils.exceptionIfNan(validation2)
	print('data 2 ready with train:', train2.shape, 'and test:', test2.shape, 'and validation:', 
	validation2.shape)
	if(SAVE_INPUT):
		data.saveData(train2, 'train2.xlsx')

if (TRAIN_MODEL_3):
	train3, test3, validation3 = data.getFinalData(INPUT_TITLE)
	train3, test3, validation3 = data.prepareMultipleData(train3, test3, validation3, [14, 15, 16])
	pred3 = data.getTestData([14, 15, 16])
	xr3 = train3[:,0:14]
	yr3 = train3[:,14:15]
	xt3 = test3[:,0:14]
	yt3 = test3[:,14:15]
	xv3 = validation3[:,0:14]
	yv3 = validation3[:,14:15]
	utils.exceptionIfNan(train3)
	utils.exceptionIfNan(test3)
	utils.exceptionIfNan(validation3)
	print('data 3 ready with train:', train3.shape, 'and test:', test3.shape, 'and validation:', 
	validation3.shape)
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
	model1.compile(loss='mean_squared_error', optimizer='adam',
	metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
	model1.fit(xr1, yr1, epochs=NUM_EPOCHS, batch_size=32, validation_data=(xv1, yv1), verbose=1)

	_, accuracy_test_1 = model1.evaluate(xt1, yt1)
	print('\n\nmodel 1 trained')
	print('\nAccuracy on test data: %.2f' % (accuracy_test_1))

	test_predictions = np.around(model1.predict(xt1), 1)

	if (PRINT_PRDECTIONS):
		for i in range(20):
			print(' test: predicted:', test_predictions[i], 'real data:', yt1[i])

	if (SAVE_OUTPUT):
		result = np.concatenate([xt1, yt1, test_predictions], axis=1)
		data.saveData(result, OUTPUT_TITLE)

	if(SAVE_WEIGHTS):
		print('\nsaving model weights ...')
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
	model2.compile(loss='mean_squared_error', optimizer='adam',
	metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
	model2.fit(xr2, yr2, epochs=NUM_EPOCHS, batch_size=32, validation_data=(xv2, yv2), verbose=1)

	_, accuracy_test_2 = model2.evaluate(xt2, yt2)
	print('\n\nmodel 2 trained')
	print('\nAccuracy on test data: %.2f' % (accuracy_test_2))

	test_predictions = np.around(model2.predict(xt2), 1)

	if (PRINT_PRDECTIONS):
		for i in range(20):
			print(' test: predicted:', test_predictions[i], 'real data:', yt2[i])

	if (SAVE_OUTPUT):
		result = np.concatenate([xt2, yt2, test_predictions], axis=1)
		data.saveData(result, OUTPUT_TITLE)

	if(SAVE_WEIGHTS):
		print('\nsaving model weights ...')
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
	model3.compile(loss='mean_squared_error', optimizer='adam', 
	metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
	model3.fit(xr3, yr3, epochs=NUM_EPOCHS, batch_size=64, validation_data=(xv3, yv3), verbose=1)

	_, accuracy_test_3 = model3.evaluate(xt3, yt3)
	print('\n\nmodel 3 trained')
	print('\nAccuracy on test data: %.2f' % (accuracy_test_3))

	test_predictions = np.around(model3.predict(xt3), 1)

	if (PRINT_PRDECTIONS):
		for i in range(20):
			print(' test: predicted:', test_predictions[i], 'real data:', yt3[i])

	if (SAVE_OUTPUT):
		result = np.concatenate([xt3, yt3, test_predictions], axis=1)
		data.saveData(result, OUTPUT_TITLE)

	if(SAVE_WEIGHTS):
		print('\nsaving model weights ...')
		output_dir = os.path.join(os.getcwd(), "saved_wights")
		model3.save_weights(filepath=os.path.join(output_dir, "all_data1_3.h5"))
		print('model weights saved')
