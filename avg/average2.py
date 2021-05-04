import numpy as np
import pandas as pd
import data as da

data = pd.read_excel('avg/duplicates.xlsx', header=0)
data = np.array(data)
final = np.zeros((250, data.shape[1]))
f = 0

ids = [False for i in range(data.shape[0])]

for i in range(data.shape[0]):
    if (ids[i]):
        continue
    a = data[i, 0:14]
    counter = 0
    col1 = 0
    col2 = 0
    col3 = 0
    tmp = np.array([])
    for j in range(data.shape[0]):
        if (i >= j):
            continue
        b = data[j, 0:14]
        if(np.all(a == b)):
            col1 = col1 + data[j, 14]
            if not(np.isnan(data[j, 15])):
                col2 = col2 + data[j, 15]
            if not(np.isnan(data[j, 16])):
                col3 = col3 + data[j, 16]
            ids[j] = True
            counter = counter + 1
    col1 = col1 + data[j, 14]
    if not(np.isnan(data[j, 15])):
        col2 = col2 + data[j, 15]
    if not(np.isnan(data[j, 16])):
        col3 = col3 + data[j, 16]
    counter = counter + 1
    final[f, :] = data[i, :]
    final[f, 14] = np.around(col1 / counter, 0)
    final[f, 15] = np.around(col2 / counter, 1)
    final[f, 16] = np.around(col3 / counter, 1)
    f = f + 1
    ids [i] = True

da.saveData(final, 'avg/duplicates_avg.xlsx')