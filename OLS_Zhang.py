# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 07:59:15 2021

@author: Zhang Lianfa
"""
import numpy as np
import numpy.linalg as la

class OLS(object):
    version = 1.0
    def __init__(self, X,y):
        self.X = np.array(X)
        
        self.y = np.array(y)
        self.betas=None
    def fit(self):
        #for OLS  model
        self.betas=self.compute_betas(self.y,self.X)
        return self.betas

    def compute_betas(self,y, x):
        xT = x.T
        xtx = np.dot(xT, x)
        xtx_inv = la.inv(xtx)
        xtx_inv_xt=np.dot(xtx_inv, xT)
        betas = np.dot(xtx_inv_xt, y)
        return betas
    def predict(self,x):
        x=np.array(x)
        yhat=x.dot(self.betas)
        return np.array(yhat)
    

import pandas as pd
data = pd.read_excel(r'./data/Tokyo/Tokyomortality.xls',0) 


#ZLF的OLS模型，见OLS_Zhang.py模块，只是简单的实现了回归算法，没有实现任何统计功能
import OLS_Zhang


filename = "data_singlevar.txt" #sys.argv[1]
X = []
y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)

X = np.array(X).reshape((len(X),1))
X = np.hstack([np.ones((len(X), 1)), X])
y = np.array(y)

OLSModel = OLS_Zhang.OLS(X,y)
OLSModel.fit()
y_pred_Zhang=OLSModel.predict(X)
print(OLSModel.betas)
#y_pred_Zhang1=OLSModel.predict([[25,13]])

import matplotlib.pyplot as plt

plt.scatter(X[:,1], y, color='green')
plt.plot(X[:,1], y_pred_Zhang, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()
