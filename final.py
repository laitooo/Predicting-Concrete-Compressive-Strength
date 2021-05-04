import data
from utils import utils
import numpy as np
import tensorflow as tf
from keras.models import Sequential

tf.compat.v1.disable_eager_execution()


NUM_EPOCHS = 15000
SAVE_INPUT = False
SAVE_OUTPUT = True
INPUT_TITLE = 'amj_avg3.xlsx'
OUTPUT_TITLE = 'amj_avg4.xlsx'
TRAIN_MODEL_1 = False
TRAIN_MODEL_2 = False
TRAIN_MODEL_3 = True


if (TRAIN_MODEL_1):
	train1, test1 = data.getFinalData(INPUT_TITLE)
	train1, test1 = data.prepareData(train1, test1, [16,17])
	print('\n train1:', train1.shape)
	xr1 = train1[:,0:14]
	yr1 = train1[:,14:16]
	xt1 = test1[:,0:14]
	yt1 = test1[:,14:16]
	utils.exceptionIfNan(train1)
	utils.exceptionIfNan(test1)
	print('data 1 ready with train:', train1.shape, 'and test:', test1.shape)
	if(SAVE_INPUT):
		data.saveData(train1, 'train1.xlsx')

if (TRAIN_MODEL_2):
	train2, test2 = data.getFinalData(INPUT_TITLE)
	train2, test2 = data.prepareData(train2, test2, [14, 15, 17])
	xr2 = train2[:,0:14]
	yr2 = train2[:,14:15]
	xt2 = test2[:,0:14]
	yt2 = test2[:,14:15]
	utils.exceptionIfNan(train2)
	utils.exceptionIfNan(test2)
	print('data 2 ready with train:', train2.shape, 'and test:', test2.shape)
	if(SAVE_INPUT):
		data.saveData(train2, 'train2.xlsx')

if (TRAIN_MODEL_3):
	train3, test3 = data.getFinalData(INPUT_TITLE)
	train3, test3 = data.prepareData(train3, test3, [14, 15, 16])
	xr3 = train3[:,0:14]
	yr3 = train3[:,14:15]
	xt3 = test3[:,0:14]
	yt3 = test3[:,14:15]
	utils.exceptionIfNan(train3)
	utils.exceptionIfNan(test3)
	print('data 3 ready with train:', train3.shape, 'and test:', test3.shape)
	if(SAVE_INPUT):
		data.saveData(train3, 'train3.xlsx')



model1 = utils.newSeqentialModel(14, 2)
model2 = utils.newSeqentialModel(14, 1)
model3 = utils.newSeqentialModel(14, 1)







if (TRAIN_MODEL_1):
	print('\n\nTrainning model 1')
	model1.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
	model1.fit(xr1, yr1, epochs=NUM_EPOCHS, batch_size=32, validation_data=(xt1, yt1), verbose=1)

	_, accuracy_test_1 = model1.evaluate(xt1, yt1)
	print('model 1 trained')
	print('Accuracy on test data: %.2f' % (accuracy_test_1*100))

	test_predictions = np.around(model1.predict(xt1), 1)

	for i in range(20):
		print(' test: predicted:', test_predictions[i], 'real data:', yt1[i])

	if (SAVE_OUTPUT):
		result = np.concatenate([test_predictions, yt1], axis=1)
		data.saveData(result, OUTPUT_TITLE)




if (TRAIN_MODEL_2):
	print('\n\nTrainning model 2')
	model2.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
	model2.fit(xr2, yr2, epochs=NUM_EPOCHS, batch_size=32, validation_data=(xt2, yt2), verbose=1)

	_, accuracy_test_2 = model2.evaluate(xt2, yt2)
	print('model 2 trained')
	print('Accuracy on test data: %.2f' % (accuracy_test_2*100))

	test_predictions = np.around(model2.predict(xt2), 1)

	for i in range(20):
		print(' test: predicted:', test_predictions[i], 'real data:', yt2[i])

	if (SAVE_OUTPUT):
		result = np.concatenate([test_predictions, yt2], axis=1)
		data.saveData(result, OUTPUT_TITLE)




if (TRAIN_MODEL_3):
	print('\n\nTrainning model 3')
	model3.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
	model3.fit(xr3, yr3, epochs=NUM_EPOCHS, batch_size=64, validation_data=(xt3, yt3), verbose=1)

	_, accuracy_test_3 = model3.evaluate(xt3, yt3)
	print('model 3 trained')
	print('Accuracy on test data: %.2f' % (accuracy_test_3*100))

	test_predictions = np.around(model3.predict(xt3), 1)

	for i in range(20):
		print(' test: predicted:', test_predictions[i], 'real data:', yt3[i])

	if (SAVE_OUTPUT):
		result = np.concatenate([test_predictions, yt3], axis=1)
		data.saveData(result, OUTPUT_TITLE)