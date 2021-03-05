import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.initializers as tfInt
from tf_utils import create_placeholders2, forward_propagation_for_predict, predict, random_mini_batches

def checkIfThereIsNan(array):
    for i in range(array.shape[0]):
        if(np.isnan(array[i]).any(axis=0)):
            return True
    return False

def plotSingleRow(array, ylabel):
    plt.figure()
    plt.plot(array)
    plt.ylabel(ylabel)
    plt.show()

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def backwardSigmoid(Z):
    return (sigmoid(Z)* (1 - sigmoid(Z)))

def forwadProbagationStep(x, w1, b1, w2, b2, w3, b3):
    Z1 = x.dot(w1) + b1
    A1 = sigmoid(Z1)
    Z2 = A1.dot(w2) + b2
    A2 = sigmoid(Z2)
    Z3 = A2.dot(w3) + b3
    #A3 = sigmoid(Z3)
    A3 = Z3
    return A3, Z3, A2, Z2, A1, Z1

def forwadProbagationStep2(x, parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    w3 = parameters['w3']
    b3 = parameters['b3']

    print('W1[0,0]:', w1[0,0])

    Z1 = tf.matmul(x, w1) + b1
    A1 = tf.nn.sigmoid(Z1)
    Z2 = tf.matmul(A1, w2) + b2
    A2 = tf.nn.sigmoid(Z2)
    Z3 = tf.matmul(A2, w3) + b3

    return Z3

def costFunction(y_real, y_generated):
    m = y_real.shape[0]
    cost = (-y_real.dot(np.log(y_generated.T)) - (1 - y_real).dot(np.log(1 - y_generated).T))
    cost = (1 / m) * np.sum(cost)
    #print('cost:', cost, 'A_generated: ', y_generated, 'y:', y_real)
    return cost

def costFunction2(y_real, y_generated):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = tf.transpose(y_generated),
        labels = tf.transpose(y_real)))
    return cost

def get_accuracy_value(y_real, y_generated):
    y_temp = y_real == y_generated
    return np.count_nonzero(y_temp)

def backwardPropagationStep(y_train, A3, Z3, w3, A2, Z2, w2, A1, Z1, x_train):
    m = y_train.shape[0]
    delta3 = (y_train - A3)
    #delta3 = (y_train - A3)
    delta2 = np.multiply(np.multiply(delta3.dot(w3.T), backwardSigmoid(A2)), Z2)
    delta1 = np.multiply(np.multiply(delta2.dot(w2.T), backwardSigmoid(A1)), Z1)

    dw3 = A2.T.dot(delta3)/m
    db3 = (np.ones((m, 1))/m).T.dot(delta3)
    dw2 = A1.T.dot(delta2)/m
    db2 = (np.ones((m, 1))/m).T.dot(delta2)
    dw1 = x_train.T.dot(delta1)/m
    db1 = (np.ones((m, 1))/m).T.dot(delta1)
    return dw3, db3, dw2, db2, dw1, db1

def initialize_parameters():
    with tf.compat.v1.Session() as sess:
        w1 = tf.compat.v1.get_variable("w1", [6, 10], initializer = tfInt.glorot_normal(seed=1))
        b1 = tf.compat.v1.get_variable("b1", [1, 10], initializer = tf.zeros_initializer())
        w2 = tf.compat.v1.get_variable("w2", [10, 5], initializer = tfInt.glorot_normal(seed=1))
        b2 = tf.compat.v1.get_variable("b2", [1, 5], initializer = tf.zeros_initializer())
        w3 = tf.compat.v1.get_variable("w3", [5, 2], initializer = tfInt.glorot_normal(seed=1))
        b3 = tf.compat.v1.get_variable("b3", [1, 2], initializer = tf.zeros_initializer())

    parameters = {"w1": w1,
                "b1": b1,
                "w2": w2,
                "b2": b2,
                "w3": w3,
                "b3": b3}

    return parameters


def model(x_train, y_train, x_test, y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    #tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(5)
    (_, n_x) = x_train.shape
    n_y = y_train.shape[1]                            # n_y : output size
    costs = []
    seed = 4                                          # to keep consistent results

    
    X, Y = create_placeholders2(n_x, n_y)
    parameters = initialize_parameters()
    Z3 = forwadProbagationStep2(X, parameters)
    cost = costFunction2(Y, Z3)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            #num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(x_train, y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer,cost],feed_dict = {X:minibatch_X, Y:minibatch_Y})
                epoch_cost += minibatch_cost / minibatch_size
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        print(parameters['w1'])
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy:", accuracy.eval({X: x_train, Y: y_train}))
        print ("Test Accuracy:", accuracy.eval({X: x_test, Y: y_test}))
        
        return parameters



