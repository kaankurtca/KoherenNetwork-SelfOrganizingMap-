import numpy as np

class Coheren():
    def __init__(self, non, dim_X ):
        self.dim_X=dim_X
        self.non=non
        self.neuronLoc=np.random.randint(0,10,[self.non, 2])
        k=0
        for i in range(int(np.sqrt(self.neuronLoc.shape[0]))):
            for j in range(int(np.sqrt(self.neuronLoc.shape[0]))):
                self.neuronLoc[k, :]= dim_X*i, dim_X*j
                k+=1
        self.neuronWeights=1*np.random.rand(non,dim_X)-0.5

    def neuronDist(self,i,j):
        return np.linalg.norm(self.neuronLoc[i]-self.neuronLoc[j])

    def competition(self,X):
        products=np.zeros(self.non)
        for ind,w in enumerate(self.neuronWeights):
            products[ind]=np.dot(w,X)
        self.winner=np.argmax(products)
        return self.winner

    def train(self,X_train,epochs,lr,sigma):

        for n in range(epochs):

            index=np.random.randint(0,len(X_train))

            i = self.competition(X_train[index])

            for j in range(len(self.neuronWeights)):
                self.neuronWeights[j] = self.neuronWeights[j] + (lr*np.exp(-n/1000)*np.exp((-self.neuronDist(i,j)**2)/(2*(sigma*np.exp(-n*np.log(sigma)/1000))**2)))*(X_train[index]-self.neuronWeights[j])











