# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:35:50 2016

@author: 19514733
"""

import numpy as np
#from gen_iso_regn import GeneralisedIsotonicRegression
from partial_mt_instance import GeneralisedIsotonicRegression
import matplotlib.pyplot as plt
#X=np.arange(20)
n=20
y=np.arange(n)
y[3] =1.5
y[6]=10
y[14]=5
y[18]=25

weights=np.ones(n)
weights[14]=10
#constraints=[]
#for i in X:
#    if i<X.shape[0]-1:
#        constraints.append([i,i+1])
constraints=[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],[16,17]] # default to monotone increasing
regn=GeneralisedIsotonicRegression()
y_iso=regn.fit(y,constraints,sample_weight=weights,increasing=True)

pseudo_X=np.arange(len(y))
plt.plot(pseudo_X,y)
plt.plot(pseudo_X,y_iso)
plt.legend(['data','iso regn'],loc=2)