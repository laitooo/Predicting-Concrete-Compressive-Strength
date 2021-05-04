import numpy as np
import pandas as pd
import data as da

data = pd.read_excel('our_data2.xlsx', header=0)
data = np.array(data)
data = np.delete(data , 14, 1)
normal = np.zeros(data.shape)
final = np.zeros(data.shape)
n = 0
f = 0

ids = [False for i in range(data.shape[0])]

for i in range(data.shape[0]):
    if (ids[i]):
        continue
    a = data[i, 0:14]
    tmp = False
    for j in range(data.shape[0]):
        if (i >= j):
            continue
        b = data[j, 0:14]
        if(np.all(a == b)):
            final[f, :] = data[j, :]
            f = f + 1
            ids[j] = True
            tmp = True
    if (tmp):
        final[f, :] = data[i, :]
        f = f + 1
        ids[i] = True
    else:
        ids[i] = False

for i in range(data.shape[0]):
    if not (ids[i]):
        normal[n, :] = data[i, :]
        n = n + 1

da.saveData(normal, 'avg/not_duplicates.xlsx')
da.saveData(final, 'avg/duplicates.xlsx')