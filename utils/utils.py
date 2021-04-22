import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import LayerNormalization
import os
import tensorflow as tf


def checkIfThereIsNan(array):
    for i in range(array.shape[0]):
        if(np.isnan(array[i]).any(axis=0)):
            return True
    return False

def exceptionIfNan(array):
    for i in range(array.shape[0]):
        if(np.isnan(array[i]).any(axis=0)):
            raise('\n\nThere is a Nan value in data\n\n')

def plotSingleRow(array, ylabel):
    plt.figure()
    plt.plot(array)
    plt.ylabel(ylabel)
    plt.show()

def newSeqentialModel(input, output):
    model = Sequential()
    model.add(LayerNormalization(input_dim=input))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(80, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(20, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(output))
    return model


def keras_to_tensorflow(keras_model, output_dir, model_name,out_prefix="output_", log_tensorboard=False):

    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)

    out_nodes = []

    print('**********************************************************************************')
    for i in range(len(keras_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(keras_model.output[i], out_prefix + str(i + 1))
        print(out_prefix + str(i + 1))
    print('**********************************************************************************')

    sess = tf.compat.v1.keras.backend.get_session()

    from tensorflow.python.framework import graph_util, graph_io
    

    init_graph = sess.graph.as_graph_def()
    main_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)

    graph_io.write_graph(main_graph, output_dir, name='saved_model.pb', as_text=False)

    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard

        import_pb_to_tensorboard.import_to_tensorboard(
            output_dir,
            output_dir,
            'concrete')