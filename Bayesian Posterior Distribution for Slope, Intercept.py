# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 08:19:50 2024

Bayesian Linear Regression (Alternate Approach)

@author: SROTOSHI GHOSH 
"""

import numpy as np
import pandas as pd
import my_linreg as mlr
import matplotlib.pyplot as plt
import scipy.stats as sc

#loading the linear regression data
cl=3*10**8
df=pd.read_csv('data 1.csv')
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

#from analytical calculations, initiating: 
m_b=0
s_b=1.8*10**(-7)
m_a=26.753
s_a=12.536

s_bf1=1/np.sqrt((1/s_b**2)+(np.sum(exi)/var))
m_bf1=((s_bf1**2)/(s_b**2))*m_b + (np.sum(exi)/var)*(s_bf1**2)*m
s_af1=1/np.sqrt((1/s_a**2)+(N/var))
m_af1=((s_af1**2)/(s_a**2))*m_a + (N/var)*(s_af1**2)*ymean

tol=0.0001
count=0

#determining means and standard deviations of posteriors in each slope and intercept
for i in range (0,1000):
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


x1=np.linspace(m_bf-4*s_bf,m_bf+4*s_bf,1000)
y1=sc.norm.pdf(x1,m_bf,s_bf)
x1_1=np.linspace(m_bf1-4*s_bf1,m_bf1+4*s_bf1,1000)
y1_1=sc.norm.pdf(x1_1,m_bf1,s_bf1)

x2=np.linspace(m_af-4*s_af,m_af+4*s_af,1000)
y2=sc.norm.pdf(x2,m_af,s_af)
x2_1=np.linspace(m_af1-4*s_af1,m_af1+4*s_af1,1000)
y2_1=sc.norm.pdf(x2_1,m_af1,s_af1)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# Plot the first normal distribution
axs[0].plot(x1, y1)
axs[0].set_title('Posterior Distribution for Slope')
axs[0].grid(True)

# Plot the second normal distribution
axs[1].plot(x2, y2)
axs[1].set_title('Posterior Distribution for Intercept')
axs[1].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

#Bayesian Credible Intervals for Slope, Intercept
ci_b=mlr.newton_raphson(0.05,N-2)*s_bf
print ("95% credible interval of slope : ",m_bf," +/- ",ci_b)
ci_a=mlr.newton_raphson(0.05,N-2)*s_af
print ("95% credible interval of slope : ",m_af," +/- ",ci_a)


                  