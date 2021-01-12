import numpy as np
import matplotlib.pyplot as plt
from SOM import Coheren
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd
import seaborn as sn

my_data = np.genfromtxt('Iris.csv', delimiter=',')     #İris veriseti numpy array'ine dönüştürüldü.
dataset=my_data[1:,1:5]       # verisetinin özellikleri(feature) kısmını ayıkladık.


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.suptitle("Iris veriseti 3 farklı özelliği ile çizdirildi.")
ax.scatter(dataset[:50,0],dataset[:50,1],dataset[:50,2],c='r',label="Iris-Setosa")
ax.scatter(dataset[50:100,0],dataset[50:100,1],dataset[50:100,2],c='b',label="Iris-Versicolor")
ax.scatter(dataset[100:,0],dataset[100:,1],dataset[100:,2],c='g',label="Iris-Virginica")
ax.set_xlabel("SepalLength"); ax.set_ylabel("SepalWidth"); ax.set_zlabel("PetalLength")
plt.legend(loc="upper right")

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
fig2.suptitle("Iris veriseti 3 farklı özelliği ile çizdirildi.")
ax2.scatter(dataset[:50,1],dataset[:50,2],dataset[:50,3],c='r',label="Iris-Setosa")
ax2.scatter(dataset[50:100,1],dataset[50:100,2],dataset[50:100,3],c='b',label="Iris-Versicolor")
ax2.scatter(dataset[100:,1],dataset[100:,2],dataset[100:,3],c='g',label="Iris-Virginica")
ax2.set_xlabel("SepalWidth"); ax2.set_ylabel("PetalLength"); ax2.set_zlabel("PetalWidth")
plt.legend(loc="upper right")

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
fig3.suptitle("Iris veriseti 3 farklı özelliği ile çizdirildi.")
ax3.scatter(dataset[:50,0],dataset[:50,2],dataset[:50,3],c='r',label="Iris-Setosa")
ax3.scatter(dataset[50:100,0],dataset[50:100,2],dataset[50:100,3],c='b',label="Iris-Versicolor")
ax3.scatter(dataset[100:,0],dataset[100:,2],dataset[100:,3],c='g',label="Iris-Virginica")
ax3.set_xlabel("SepalLength"); ax3.set_ylabel("PetalLength"); ax3.set_zlabel("PetalWidth")
plt.legend(loc="upper right")

scaler = StandardScaler()
dataset = scaler.fit_transform(dataset) # öbeklemenin daha iyi olması için verisetinin sütunları ayrı ayrı standardize edildi.

randomIndex_1=np.random.choice(50,40,replace=False).reshape(1,40)
randomIndex_2=50+np.random.choice(50,40,replace=False).reshape(1,40)
randomIndex_3=100+np.random.choice(50,40,replace=False).reshape(1,40)   # Her sınıftan rastgele 40 örnek seçilecek şekilde indis seçildi
temp=np.concatenate([randomIndex_1,randomIndex_2],axis=0)
trainSet_Index=np.concatenate([temp,randomIndex_3],axis=0).reshape(120,) # Seçilen indisler birleştirildi, bir arrayde tutuldu.

trainSet=dataset[trainSet_Index,:]  #her sınıftan rastgele 40'ar tane seçilerek, 120 veriden oluşan eğitim kümesi oluşturuldu.  # 120x7
testSet=np.delete(dataset,trainSet_Index,axis=0)    # Test kümesi oluşturuldu.


som = Coheren(9,4) # Noron sayısı:16 Veri boyutu:4
fig4 = plt.figure()
plt.scatter(som.neuronLoc[:,0],som.neuronLoc[:,1],) #Nöronların fiziksel konumları çizdirildi.
plt.title("Nöronların fiziksel konumları")




print("\n\nNöronların ilk ağırlıkları: ",som.neuronWeights)
som.ordering(trainSet,1000,0.1,5.65)  # Eğitimin ilk aşaması: özdüzenleme (self-organizing,ordering)
som.convergence(trainSet,500*9,0.01,5.65)    # Eğitimin ikinci aşaması: yakınsama (convergence)

sınıflar=np.zeros(30)
sınıflar[10:20],sınıflar[20:]=1,2   # karşılaştırma yapabilmek için verilerin gerçek sınıfları oluşturuldu.

comparison=np.zeros((len(testSet), 2))
print("\nTahmin edilen öbekler ve Gerçek sınıflar (aynı sınıf için aynı öbekler bekleniyor.)")
for i,data in enumerate(testSet):
    comparison[i, 0] = som.competition(data)
    comparison[i, 1] = sınıflar[i]
    # comparison matrisinin ilk sütunu tahminler( yani kazanan nöronlar), ikinci sütunu gerçek sınıflar.
    # burdaki beklentimiz sayıların birebir aynı olması değil ama her sınıfta farklı bir nöronun kazanmış olması.

print(comparison)

print("\n\nNöronların son ağırlıkları: ",som.neuronWeights)


first = stats.mode(comparison[0:10,0])
comparison[:10,0]=np.where(comparison[:10,0]==first[0][0],0,comparison[:10,0])
second = stats.mode(comparison[10:20,0])
comparison[10:20,0]=np.where(comparison[10:20,0]==second[0][0],1,comparison[10:20,0])
third = stats.mode(comparison[10:,0])
comparison[20:,0]=np.where(comparison[20:,0]==third[0][0],2,comparison[20:,0])
# Burada, confusion matrixte kıyaslama yapabilmemiz için her sınıfın en çok seçtiği nöronu sırasıyla 0,1,2 olarak değiştiriyoruz.

from sklearn import metrics
confusionMatrix=metrics.confusion_matrix(comparison[:,1], comparison[:,0],labels=[0,1,2])
df = pd.DataFrame(confusionMatrix, range(3), range(3))
fig4 = plt.figure()
fig4.suptitle("Confusion Matrix\n(Her sınıftan 10 veri var.)")
sn.set(font_scale=1.4)
sn.heatmap(df, annot=True, annot_kws={"size": 16})

plt.show()
