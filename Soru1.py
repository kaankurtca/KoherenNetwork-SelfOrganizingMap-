import numpy as np
import matplotlib.pyplot as plt
from SOM import Coheren

a=np.random.normal(0.7,0.5,[50,3]) - np.array([0,0.80,4])
b=np.random.normal(0.1,0.5,[50,3]) - np.array([-3,-0.6,-1])
c=np.random.normal(0.4,0.5,[50,3]) - np.array([1,-2,0.5])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(a[:,0],a[:,1],a[:,2],c='r')
ax.scatter(b[:,0],b[:,1],b[:,2],c='b')
ax.scatter(c[:,0],c[:,1],c[:,2],c='g')
plt.show()

temp=np.concatenate([a,b],axis=0)
dataset=np.concatenate([temp,c],axis=0)

som = Coheren(9,3)
print("\n\nNöronların ilk ağırlıkları: ",som.neuronWeights)
som.train(dataset,15000,0.01,10)

sınıflar=np.zeros(150)
sınıflar[50:100],sınıflar[100:]=1,2
for i,data in enumerate(dataset):
    tahmin = som.competition(data)
    gercek = sınıflar[i]
    print("[ {}  {} ]".format(tahmin, int(gercek)))

print("\n\nNöronların fiziksel konumları: ",som.neuronLoc)
print("\n\nNöronların son ağırlıkları: ",som.neuronWeights)


plt.show()