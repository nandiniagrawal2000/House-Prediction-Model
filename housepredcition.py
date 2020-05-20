import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def normalequation(X,y):
    theta=np.zeros((X.shape[1],1))
    theta1=np.linalg.inv(np.dot(X.transpose(),X))
    theta=np.dot(theta1,np.dot(X.transpose(),y))
    return theta

def costfunction(theta,X,y):
    m=y.shape[0]
    J=np.dot((np.dot(X,theta)-y).transpose(),(np.dot(X,theta)-y))
    return J/(2*m)

def gradientdescent(theta,X,y,iterations):
    J_history=np.zeros(iterations)
    m=y.shape[0]
    alpha=0.3
    for iter in range(iterations):
        theta=theta-np.dot(X.transpose(),np.dot(X,theta)-y)*alpha/m
        J_history[iteration]=costfunction(theta,X,y)

    return theta,J_history


def mean(X):
       return sum(X)/X.size

def featurise(X):
    mu=np.zeros((1,X.shape[1]))
    diff=np.zeros((1,X.shape[1]))
    X_norm=X
    for i in range(X.shape[1]):
        mu[0][i]=mean(X[:,i:i+1])
        diff[0][i]=max(X[:,i:i+1])-min(X[:,i:i+1])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_norm[i][j]=(X[i][j]-mu[0][j])/diff[0][j]


    return X_norm

#col_list=["size","number","price"]
a=open("house.txt")
a1=a.readlines()
count=0
for i in a1:
    count+=1
x=np.zeros((count,2),dtype="float64")
y=np.zeros((count,1),dtype="int64")
for i in range(count):
    size,number,price=a1[i].split(",")
    x[i][0]=int(size)
    x[i][1]=int(number)
    y[i][0]=int(price)

q=np.ones((count,1),dtype="float64")
x=featurise(x)
X=np.concatenate((q,x),axis=1)
#print(X.shape[1])

theta=np.zeros((X.shape[1],1))
theta,J_history=gradientdescent(theta,X,y,1500)

print(theta)

print(normalequation(X,y))

plt.plot(J_history)
plt.show()
