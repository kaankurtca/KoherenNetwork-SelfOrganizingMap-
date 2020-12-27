import numpy as np
import matplotlib.pyplot as plt
from SOM import Coheren

a=np.random.normal(1,0.2,[50,3]) - np.array([4,0,0])
b=np.random.normal(1,0.2,[50,3]) - np.array([-3,2,1])
c=np.random.normal(1,0.2,[50,3]) - np.array([0,0,3])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(a[:,0],a[:,1],a[:,2],c='r')
ax.scatter(b[:,0],b[:,1],b[:,2],c='b')
ax.scatter(c[:,0],c[:,1],c[:,2],c='g')


temp=np.concatenate([a,b],axis=0)
dataset=np.concatenate([temp,c],axis=0)

som = Coheren(9,3)

som.train(dataset,5000,0.1,10)


for data in dataset:
    print(som.competition(data))

plt.show()