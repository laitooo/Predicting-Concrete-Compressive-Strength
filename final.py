import data
from utils import utils
import numpy as np
import tensorflow as tf
from keras.models import Sequential

tf.compat.v1.disable_eager_execution()


NUM_EPOCHS = 2000


train1, test1 = data.getFinalData()
train1, test1 = data.prepareData(train1, test1, [16,17])
print('\n train1:', train1.shape)
xr1 = train1[:,0:14]
yr1 = train1[:,14:16]
xt1 = test1[:,0:14]
yt1 = test1[:,14:16]
utils.exceptionIfNan(train1)
utils.exceptionIfNan(test1)
print('data 1 ready with train:', train1.shape, 'and test:', test1.shape)

train2, test2 = data.getFinalData()
train2, test2 = data.prepareData(train2, test2, [14, 15, 17])
xr2 = train2[:,0:14]
yr2 = train2[:,14:15]
xt2 = test2[:,0:14]
yt2 = test2[:,14:15]
utils.exceptionIfNan(train2)
utils.exceptionIfNan(test2)
print('data 2 ready with train:', train2.shape, 'and test:', test2.shape)

train3, test3 = data.getFinalData()
train3, test3 = data.prepareData(train3, test3, [14, 15, 16])
xr3 = train3[:,0:14]
yr3 = train3[:,14:15]
xt3 = test3[:,0:14]
yt3 = test3[:,14:15]
utils.exceptionIfNan(train3)
utils.exceptionIfNan(test3)
print('data 3 ready with train:', train3.shape, 'and test:', test3.shape)
data.saveData(train3, 'train3.xlsx')



model1 = utils.newSeqentialModel(14, 2)
model2 = utils.newSeqentialModel(14, 1)
model3 = utils.newSeqentialModel(14, 1)








print('\n\nTrainning model 1')
model1.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
model1.fit(xr1, yr1, epochs=NUM_EPOCHS, batch_size=32, validation_data=(xt1, yt1), verbose=1)

_, accuracy_test_1 = model1.evaluate(xt1, yt1)
print('model 1 trained')
print('Accuracy on test data: %.2f' % (accuracy_test_1*100))

test_predictions = np.around(model1.predict(xt1), 1)

for i in range(20):
	print(' test: predicted:', test_predictions[i], 'real data:', yt1[i])





print('\n\nTrainning model 2')
model2.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
model2.fit(xr2, yr2, epochs=NUM_EPOCHS, batch_size=32, validation_data=(xt2, yt2), verbose=1)

_, accuracy_test_2 = model2.evaluate(xt2, yt2)
print('model 1 trained')
print('Accuracy on test data: %.2f' % (accuracy_test_2*100))

test_predictions = np.around(model2.predict(xt2), 1)

for i in range(20):
	print(' test: predicted:', test_predictions[i], 'real data:', yt2[i])






print('\n\nTrainning model 3')
model3.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
model3.fit(xr3, yr3, epochs=NUM_EPOCHS, batch_size=64, validation_data=(xt3, yt3), verbose=1)

_, accuracy_test_3 = model3.evaluate(xt3, yt3)
print('model 1 trained')
print('Accuracy on test data: %.2f' % (accuracy_test_3*100))

test_predictions = np.around(model3.predict(xt3), 1)

for i in range(20):
	print(' test: predicted:', test_predictions[i], 'real data:', yt3[i])


