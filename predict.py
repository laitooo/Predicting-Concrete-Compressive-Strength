import data
import sys
import numpy as np
from tensorflow import keras

if(len(sys.argv) < 7):
    raise ValueError('Missing input arguemnts')

model = keras.models.load_model("my_model")

# sample no. 75
#x = np.array([[360, 0.45, 160, 1880, 715, 1165]])
x = np.array([[float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), 
    float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])]])
y_hat = np.around(model.predict(x), 1)
y = np.array([[33.1, 39.1]])

print('prediction:', y_hat, ' real:', y)