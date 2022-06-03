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
# 4 cement
# 5 w_c
# 6 water
# 7 additive_type
# 8 additive_dosage
# 9 ad_solid
# 10 coarse
# 11 fine

if(len(sys.argv) > 13):
    raise ValueError('Input are more than needed', 'input are', sys.argv)
elif(len(sys.argv) < 13):
    raise ValueError('Missing input')
    
# model = keras.models.load_model("my_model")
model = utils.newSeqentialModel(12, 1)
model2 = utils.newSeqentialModel(12, 1)
model3 = utils.newSeqentialModel(12, 1)
output_dir = os.path.join(os.getcwd(), "saved_wights")
error = False

try:
    model.load_weights(filepath=os.path.join(output_dir, 'additives2_7days' + ".h5"))
    model2.load_weights(filepath=os.path.join(output_dir, 'additives2_28days' + ".h5"))
    model3.load_weights(filepath=os.path.join(output_dir, 'additives2_slump' + ".h5"))
except OSError:
    print('no previous weights found')
    error = True
except ValueError:
    print('previous weights are different from current') 
    error = True

if (not error) :
    x = np.ones((1, 12))
    for i in range(1, 12) :
        x[0][i - 1] = float(sys.argv[i])
    #x[0][11] = float(sys.argv[11])
    #x[0][12] = float(sys.argv[12])
    #x[0][10] = x[0][11] + x[0][12]

    y_hat = np.around(model.predict(x), 1)
    y_hat2 = np.around(model2.predict(x), 1)
    y_hat3 = np.around(model3.predict(x), 1)


    print('predictions :')
    print('for 7 days :', y_hat)
    print('for 28 days :', y_hat2)
    print('for slump :', y_hat3)