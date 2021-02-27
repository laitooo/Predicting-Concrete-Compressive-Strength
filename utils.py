import numpy as np
import matplotlib.pyplot as plt

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
