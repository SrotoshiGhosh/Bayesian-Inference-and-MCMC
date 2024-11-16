# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:23:09 2024

Bayesian Predictive Distribution (Alternate Approach)

@author: DELL
"""

import numpy as np
import pandas as pd
import my_linreg as mlr
import matplotlib.pyplot as plt
import scipy.stats as sc

#loading the linear regression data
cl=3*10**8
df=pd.read_csv('data.csv')
ri=df['NASA dist'] #stores distance of galaxies from earth
zi=df['NASA z'] 
vi=[] #stores velocity of recession of galaxies, evaluated from redshift 
for i in range (0,len(zi)):
    vi.append(zi[i]*cl)
   
N=len(vi)
m,er_slope,c,er_inter,xi2,xiyi,ei,ei2,exi,cov=mlr.lin_reg(vi,ri)
var=np.sum(ei2)/(N-2)
ym=[]
for i in range (0,N):
    ym.append(m*vi[i]+c)
'''  
plt.plot(vi,ri,'r.')
plt.plot(vi,ym)
'''
print ("From least squares fitting, the slope and intercept, respectively, are : ",m,c)

xmean=np.mean(vi)
ymean=m*xmean+c
print(ymean)

ym=[]
for i in range (0,N):
    ym.append(m*vi[i]+c)

#from analytical calculations, initiating: 
m_b=0
s_b=1.8*10**(-7)
m_a=26.742
s_a=12.536
tol=0.0001
count=0

#determining means and standard deviations of posteriors in each slope and intercept
for i in range (0,10000):
    s_bf=1/np.sqrt((1/s_b**2)+(np.sum(exi)/var))
    m_bf=((s_bf**2)/(s_b**2))*m_b + (np.sum(exi)/var)*(s_bf**2)*m
    
    s_af=1/np.sqrt((1/s_a**2)+(N/var))
    m_af=((s_af**2)/(s_a**2))*m_a + (N/var)*(s_af**2)*ymean
    
    if(np.abs(s_bf-s_b)<=tol and np.abs(m_bf-m_b)<=tol and np.abs(s_af-s_a)<=tol and np.abs(m_af-m_a)<=tol):
        break
    else:
        s_b=s_bf
        s_a=s_af
        m_b=m_bf
        m_a=m_af
        count+=1

print("For slope : mean = ",m_bf,"sd = ",s_bf )
print("For intercept : mean = ",m_af,"sd = ",s_af )
print("Number of iterations taken to converge : ",count)

#PredictiveDistr
xi=(0.028762)*cl #NGC7603
yi=121.74
m_pred=m_af+((xi-xmean)*m_bf)
s_pred=np.sqrt((s_af)**2+((xi-xmean)**2 *(s_bf)**2)+var)

print(" ")
print("Given x : ",(xi))
print("Observed y : ",yi)
print("The predicted value of y for given x from this model is : ",m_pred)
print(s_pred)

x=np.linspace(m_pred-4*s_pred,m_pred+4*s_pred,1000)
y=sc.norm.pdf(x,m_pred,s_pred)
plt.plot(x,y)
plt.grid()
plt.xlabel("y")
plt.ylabel("P(y|D)")
'''
plt.plot(vi,ri,'r.')
plt.plot(xi,yi,'g*')
'''
