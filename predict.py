import os
from re import L
import data
import sys
import numpy as np
from utils import utils
from tensorflow import keras

# 0 coarse_type
# 1 fine_type
# 2 max_size
# 3 passing
# 4 target_mean
# 5 grade
# 6 cement
# 7 w_c
# 8 water
# 9 additive
# 10 total
# 11 coarse
# 12 fine

if(len(sys.argv) > 13):
    raise ValueError('Input are more than needded')
elif(len(sys.argv) < 13):
    raise ValueError('Missing input')
    
# model = keras.models.load_model("my_model")
model = utils.newSeqentialModel(13, 1)
model2 = utils.newSeqentialModel(13, 1)
output_dir = os.path.join(os.getcwd(), "saved_wights")
error = False

try:
    model.load_weights(filepath=os.path.join(output_dir, 'gand0_9_12_7days' + ".h5"))
    model2.load_weights(filepath=os.path.join(output_dir, 'gand0_9_12_28days' + ".h5"))
except OSError:
    print('no previous weights found')
    error = True
except ValueError:
    print('previous weights are different from current') 
    error = True

if (not error) :
    x = np.ones((1, 13))
    for i in range(1, 11) :
        x[0][i - 1] = float(sys.argv[i])
    x[0][11] = float(sys.argv[11])
    x[0][12] = float(sys.argv[12])
    x[0][10] = x[0][11] + x[0][12]

    y_hat = np.around(model.predict(x), 1)
    y_hat2 = np.around(model2.predict(x), 1)
    y = np.array([[31.1]])


    print('predictions :')
    print('for 7 days :', y_hat)
    print('for 28 days :', y_hat2)