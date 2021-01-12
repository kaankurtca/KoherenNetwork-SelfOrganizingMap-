import numpy as np
import matplotlib.pyplot as plt
from SOM import Coheren
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd
import seaborn as sn

a=np.random.normal(3,0.5,[200,3]) - np.array([0,0.2,1])
b=np.random.normal(1,0.5,[200,3]) - np.array([1,-0.6,0])
c=np.random.normal(2,0.5,[200,3]) - np.array([0,0,2])
# farklı ortalama ve standart sapmaya sahip 3 boyutlu 3 veri kümesi oluşturuldu.


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.suptitle("3 boyutlu düzlemde verisetimiz")
ax.scatter(a[:,0],a[:,1],a[:,2],c='r')
ax.scatter(b[:,0],b[:,1],b[:,2],c='b')
ax.scatter(c[:,0],c[:,1],c[:,2],c='g')
# 3 boyutlu düzlemde çizdirilerek gözlemlendi.


temp=np.concatenate([a,b],axis=0)
dataset=np.concatenate([temp,c],axis=0) # veriseti birleştirildi.

scaler = StandardScaler()
dataset = scaler.fit_transform(dataset) # öbeklemenin daha iyi olması için verisetinin sütunları ayrı ayrı standardize edildi.

randomIndex_1=np.random.choice(200,160,replace=False).reshape(-1,1)
randomIndex_2=200+np.random.choice(200,160,replace=False).reshape(-1,1)
randomIndex_3=400+np.random.choice(200,160,replace=False).reshape(-1,1)   # Her sınıftan rastgele 160 örnek seçilecek şekilde indis seçildi
temp=np.concatenate([randomIndex_1,randomIndex_2],axis=0)
trainSet_Index=np.concatenate([temp,randomIndex_3],axis=0).reshape(-1,) # Seçilen indisler birleştirildi, bir arrayde tutuldu.

trainSet=dataset[trainSet_Index,:]  #her sınıftan rastgele 160'ar tane seçilerek, 480 veriden oluşan eğitim kümesi oluşturuldu.
testSet=np.delete(dataset,trainSet_Index,axis=0)    # Test kümesi oluşturuldu.

som = Coheren(9,3) # Noron sayısı:9 Veri boyutu:3

fig3 = plt.figure()
plt.scatter(som.neuronLoc[:,0],som.neuronLoc[:,1])  #Nöronların fiziksel konumları çizdirildi.
plt.title("Nöronların fiziksel konumları")

print("\n\nNöronların ilk ağırlıkları: ",som.neuronWeights)

som.ordering(trainSet,1000,0.1,20)  # Eğitimin ilk aşaması: özdüzenleme (self-organizing,ordering)  epochs=1000
som.convergence(trainSet,500*som.non,0.01,20)  # Eğitimin ikinci aşaması: yakınsama (convergence) epochs=500*nöronSayısı

sınıflar=np.zeros(120)
sınıflar[40:80],sınıflar[80:]=1,2 # karşılaştırma yapabilmek için verilerin gerçek sınıfları oluşturuldu.

comparison=np.zeros((len(testSet), 2))
print("\nTahmin edilen öbekler ve Gerçek sınıflar (aynı sınıf için aynı öbekler bekleniyor.)")
for i,data in enumerate(testSet):
    comparison[i, 0] = som.competition(data)
    comparison[i, 1] = sınıflar[i]
    # comparison matrisinin ilk sütunu tahminler( yani kazanan nöronlar), ikinci sütunu gerçek sınıflar.
    # burdaki beklentimiz sayıların birebir aynı olması değil ama her sınıfta farklı bir nöronun kazanmış olması.
print(comparison)


print("\n\nNöronların son ağırlıkları: ",som.neuronWeights)

print("\nNöronların ortalama ağırlıkları")
for i in range(len(som.neuronWeights)):
    ak=np.mean(som.neuronWeights[i])
    print(i,". nöron ortalama ağırlığı: ",ak)

print("\n0. küme için ortalama: ",np.mean(dataset[0:200,:]))
print("1. küme için ortalama: ",np.mean(dataset[200:400,:]))
print("2. küme için ortalama: ",np.mean(dataset[400:,:]))



first = stats.mode(comparison[0:40,0])
comparison[:40,0]=np.where(comparison[:40,0]==first[0][0],0,comparison[:40,0])
second = stats.mode(comparison[40:80,0])
comparison[40:80,0]=np.where(comparison[40:80,0]==second[0][0],1,comparison[40:80,0])
third = stats.mode(comparison[80:,0])
comparison[80:,0]=np.where(comparison[80:,0]==third[0][0],2,comparison[80:,0])
# Burada, confusion matrixte kıyaslama yapabilmemiz için her sınıfın en çok seçtiği nöronu sırasıyla 0,1,2 olarak değiştiriyoruz.

from sklearn import metrics
confusionMatrix=metrics.confusion_matrix(comparison[:,1], comparison[:,0],labels=[0,1,2])
df = pd.DataFrame(confusionMatrix, range(3), range(3))
fig4 = plt.figure()
fig4.suptitle("Confusion Matrix\n(Her sınıftan 40 veri var.)")
sn.set(font_scale=1.4)
sn.heatmap(df, annot=True, annot_kws={"size": 16})
# Örneğin öbeklerimizden birisi 2 farklı nörona yakınsıyorsa; confusion matrix sadece en çok verinin yakınsadığını kabul ediyor bu yüzden matriste performans düşük görünüyor.


plt.show()