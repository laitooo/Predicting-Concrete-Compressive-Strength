import numpy as np
import pandas as pd
import math


def checkIfThereIsNan(array):
    for i in range(array.shape[0]):
        if (pd.isna(array[i]).any(axis=0)):
            print('\nn/a data in row', i)
            return True
    return False

data = pd.read_excel('missing_data/data_19.xlsx', header=0)
data = np.array(data)
x = data[:,3:20]
y = data[:,20:22]

x = np.delete(x, 0, 1)  #removed string : type of coarse aggregate
x = np.delete(x, 1, 1)  #removed string : type of fine aggregate
x = np.delete(x, 8, 1)  #removed string : type of additive
x = np.delete(x, 8, 1)  #removed string : dose of additive
x = np.delete(x, 7, 1)  #removed string : water content


if(checkIfThereIsNan(x)):
    print('\nsome input data is missing')
else:
    print('\nall input data is good')

print('')


print('Completing data using the average')
x_2 = np.zeros(x.shape)

for i in range(x.shape[1]):
    tmp = x[:,i]
    mean = np.mean(tmp[~pd.isna(tmp)])
    correct = np.where(pd.isna(tmp), mean, tmp)
    x_2[:,i] = correct

df = pd.DataFrame(x_2)
filepath = 'missing_data/completed_data_19_mean.xlsx'
df.to_excel(filepath, index=False)
print('saved completed data to excel file (using mean method)')