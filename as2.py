#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:06:15 2018

@author: ranchal
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print('loading train')
df = pd.read_csv('P2_train.csv',header=None)
zero=df.loc[df.iloc[:,-1] == 0]
u0=np.mean(zero.iloc[:,0:-1])
cov0=np.cov(zero.iloc[:,0:-1].T)

one=df.loc[df.iloc[:,-1] == 1]
u1=np.mean(one.iloc[:,0:-1])
cov1=np.cov(one.iloc[:,0:-1].T)

cov0_inv = np.linalg.inv(cov0)
cov1_inv = np.linalg.inv(cov1)

cov0_det = np.linalg.det(cov0)
cov1_det = np.linalg.det(cov1)


var1=np.var(df.iloc[:,0])
var2=np.var(df.iloc[:,1])
var_m=(var1+var2/2)
#covi=np.cov(df.iloc[:,0:-1].T)
#
#covi=(cov0+cov1)/2

covi = np.zeros((2,2))
#covi[0,0]=var1
#covi[1,1]=var2
covi[0,0]=var_m
covi[1,1]=var_m


covi_inv = np.linalg.inv(covi)
covi_det = np.linalg.det(covi)

tf = pd.read_csv('P2_test.csv',header=None)
pred= np.zeros(len(tf))
d_fn= np.zeros(len(tf))

for k in range(0,len(tf)):
#    g0 =  - 0.5*np.log(cov0_det) + np.log(len(zero)/len(df)) - 0.5*np.dot(np.dot(( tf.iloc[k,0:-1] - u0), cov0_inv), ( tf.iloc[k,0:-1]- u0).T)
#    g1 =  - 0.5*np.log(cov1_det) + np.log(len(one)/len(df)) - 0.5*np.dot(np.dot(( tf.iloc[k,0:-1] - u1), cov1_inv), (tf.iloc[k,0:-1] - u1).T)    
#   
    g0 =  - 0.5*np.log(covi_det) + np.log(len(zero)/len(df)) - 0.5*np.dot(np.dot(( tf.iloc[k,0:-1] - u0), covi_inv), ( tf.iloc[k,0:-1]- u0).T)
    g1 =  - 0.5*np.log(covi_det) + np.log(len(one)/len(df)) - 0.5*np.dot(np.dot(( tf.iloc[k,0:-1] - u1), covi_inv), (tf.iloc[k,0:-1] - u1).T)    
    d_fn[k]=  g1-g0
    if g0<g1:
        pred[k]=1
    else:
        pred[k]=0
        
crct = 0
false1 = 0
false0 = 0
true0 = 0
true1= 0
label=tf.iloc[:,-1]
for p in range(len(label)):
    if label[p]-pred[p]==0:
        if label[p] == 0:
            true0 += 1
        else:
            true1 += 1
        crct += 1
    elif label[p]-pred[p] == -1:
        false1 += 1
    else:
        false0+= 1
confusion_matrix = np.array([[true0, false1],[false0, true1]]) 
plt.plot(d_fn)
plt.show()

#for i in six.moves.range(2):
#    Z = mlab.bivariate_normal(five[:,0:-1], five[:,-1], np.sqrt(var1),
#                              np.sqrt(var1),
#                              u[1], u[2])
#    plt.contour(five[:,0:-1],five[:,-1], Z)
#    plt.savefig(output)
