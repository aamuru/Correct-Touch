import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import copy
from NeuralNetClass import NeuralNetLog


df = pd.read_csv("data.csv",sep=",")

#from scipy.stats import itemfreq
#np.asmatrix(humid).shape[0]
#np.asmatrix(temper).shape[0]
#copy.copy([1,2,3])
print(df.values.shape)
groupwithout.ix[:,[2,3,4]]
#groupwithout.ix[:,[0:3]]
X = df.ix[:,[1,3,4,5,6,7,8,9,10]].values
X = X[:35000,:]
T = df.ix[:,[2]].values

T = T[:35000,:]
print(type(T))


classes = np.unique(T)
numclasses = len(np.unique(T))
print("numclasses")
print(numclasses)
base_class = np.array([0 for i in range(numclasses)])

new_T = np.array([base_class]*T.shape[0])

print(new_T)
print(new_T.shape)

for i in range(X.shape[0]):
    #print(T[i])
    #print(np.where(classes==T[i]))
    new_T[i][np.where(classes==T[i])] = 1
    

 

print(new_T)
print(new_T.shape)
trainnet = NeuralNetLog([X.shape[1],20,numclasses])
trainnet.train(X,new_T,ftracep=True)
#tranans,z = trainnet.use(X,retZ=True)
