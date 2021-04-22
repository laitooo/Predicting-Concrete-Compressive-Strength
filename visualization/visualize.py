import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plotSingleRow(array):
    plt.figure()
    plt.plot(array)
    plt.show()


data = pd.read_excel('final_data.xlsx')
data = np.array(data)
for i in range(0, data.shape[1]):
    plotSingleRow(data[:,i])
