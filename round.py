import numpy as np
import pandas as pd
import data as dt

print('loading the data ...')
data = pd.read_excel('data_files/our_data2.xlsx', header=1)
data = np.array(data)
p = np.zeros(data.shape)

for i in range(data.shape[1]):
    print('column', i, ' value', data[0:5,i])
    if i in [0, 1, 5, 6, 11, 12, 13, 14]:
        p[:, i] = data[:, i]
    elif i in [2, 4, 8, 9, 15]:
        p[:, i] = np.around(data[:, i])
    elif i in [7, 10]:
        p[:, i] = np.around(data[:, i], 2)
    else:
        p[:, i] = np.around(data[:, i], 1)
    
dt.saveData(p, 'data_files/our_data3.xlsx')