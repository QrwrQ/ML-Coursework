import numpy as np
data=np.loadtxt('iris.data',dtype=str,delimiter=',')
data=np.insert(data,5,values='0',axis=1)
data=np.insert(data,6,values='0',axis=1)
for i in range(len(data)):
    if data[i,4]=='Iris-setosa':
        data[i, 4] = '1'
        data[i, 5] = '0'
        data[i, 6] = '0'
    if data[i,4]=='Iris-versicolor':
        data[i, 4] = '0'
        data[i, 5] = '1'
        data[i, 6] = '0'
    if data[i,4]=='Iris-virginica':
        data[i, 4] = '0'
        data[i, 5] = '0'
        data[i, 6] = '1'

np.random.shuffle(data)

np.savetxt('iris.csv', data, fmt='%s', delimiter=',')
print(data[100,0:])