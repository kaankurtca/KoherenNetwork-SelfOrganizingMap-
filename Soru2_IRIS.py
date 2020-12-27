import numpy as np
import matplotlib.pyplot as plt
from SOM import Coheren

my_data = np.genfromtxt('Iris.csv', delimiter=',')     #İris veriseti numpy array'ine dönüştürüldü.
dataset_X=my_data[1:,1:5]       # verisetinin index ve sınıf bilgisi kısımlarını ayıkladık. Feature'lar kaldı.  #150x4 array


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dataset_X[:50,0],dataset_X[:50,1],dataset_X[:50,2],c='r')
ax.scatter(dataset_X[50:100,0],dataset_X[50:100,1],dataset_X[50:100,2],c='b')
ax.scatter(dataset_X[100:,0],dataset_X[100:,1],dataset_X[100:,2],c='g')

dataset_X = dataset_X - dataset_X.min(axis=0)
dataset_X = dataset_X / (dataset_X.max(axis=0) - dataset_X.min(axis=0))

som = Coheren(16,4)
print("\n\nNöronların ilk ağırlıkları: ",som.neuronWeights)
som.train(dataset_X,15000,0.0001,20)

sınıflar=np.zeros(150)
sınıflar[50:100],sınıflar[100:]=1,2
for i,data in enumerate(dataset_X):
    tahmin=som.competition(data)
    gercek=sınıflar[i]
    print("[ {}  {} ]".format(tahmin,int(gercek)))

print("\n\nNöronların fiziksel konumları: ",som.neuronLoc)
print("\n\nNöronların son ağırlıkları: ",som.neuronWeights)

plt.show()
