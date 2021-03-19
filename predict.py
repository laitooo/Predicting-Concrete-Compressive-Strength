from tensorflow import keras
import numpy as np
import data

model = keras.models.load_model("my_model")

# sample no. 75
x = np.array([[360, 0.45, 160, 1880, 715, 1165]])
y_hat = np.around(model.predict(x), 1)
y = np.array([[33.1, 39.1]])

print('prediction:', y_hat, ' real:', y)