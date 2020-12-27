import numpy as np

locMat=np.zeros([9,2])
# for k in range(len(locMat)):
k=0
for i in range(int(np.sqrt(locMat.shape[0]))):
    for j in range(int(np.sqrt(locMat.shape[0]))):
        locMat[k,]=i,j
        k+=1


print(locMat)
# neuronWeights=0*np.random.rand(9,3)
# c=np.random.normal(1,1,[200,3]) - np.array([5,0,0])
#
# arr=np.array([1,2,3,6,0,5,1,9])
#
# minInd=np.argmax(arr)
#
# print(minInd)
#
# X=np.random.rand(150,3)
#
# veri=X[0]
# w=neuronWeights[0]
# product=np.dot(w,veri)
