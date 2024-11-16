# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:52:42 2024

monte carlo integration
@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    m=0
    s=0.01
    y=np.exp(-((x-m)**2)/(2*s**2))/(s*np.sqrt(2*np.pi))
    return y

N=10**5 #numner of points taken
a=-10 #lower limit of integration
b=+10 #upper limit of integration
h=(b-a)/N

xi=np.random.uniform(a,b,N)
yi=f(xi)


maxy=np.max(yi)
miny=np.min(yi)
ym=np.random.uniform(miny,maxy,N)
plt.plot(xi,ym,'r.')
c=0
for i in range (0,N):
    if (ym[i]<=0):
        if (ym[i]>=yi[i]):
            c-=1
            plt.plot(xi[i],ym[i],'g.')
    if (ym[i]>0):
        if (ym[i]<=yi[i]):
            c+=1
            plt.plot(xi[i],ym[i],'g.')

A=(b-a)*(maxy-miny)
I=c*A/N
print ("Value of integration : ",I)


