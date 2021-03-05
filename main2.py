import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import utils
import data

tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()
tf.compat.v1.set_random_seed(1)

x_train, y_train, x_test, y_test, n = data.getData()


print('initializing model ...')
learning_rate = 0.03
epochs = 2000
minibatch_size = 128
    

print('started trainning ...')
parameters = utils.model(x_train, y_train, x_test, y_test, learning_rate,
          epochs, minibatch_size, print_cost = True)