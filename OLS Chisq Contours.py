# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:13:38 2024

Plots chi-squared contours for Ordinary Least Squares Fitting

@author: SROTOSHI GHOSH 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def chisq(x,y,m,c,I):
    s=0
    for i in range (0,len(x)):
        s+=(y[i]*I-m*x[i]-c*I)**2
    return s

def linreg(x,y):
    N=len(x)
    s_x=np.sum(x)
    s_y=np.sum(y)
    s_xy=0
    s_xx=0
    for i in range (0,N):
        s_xy+=(x[i]*y[i])
        s_xx+=(x[i]**2)
    #slope of line of best fit
    m=((N*s_xy)-(s_x*s_y))/((N*s_xx)-(s_x**2))
    #intercept of line of best fit
    c=(s_y-m*s_x)/N
    #errors in m and c
    ei2=[]
    for i in range (0,N):
        term=y[i]-m*x[i]-c
        ei2.append(term**2)
    
    meanx=np.mean(x)
    ex=[]
    for i in range (0,N):
        term=(x[i]-meanx)**2
        ex.append(term)
    errm=np.sqrt(np.sum(ei2)/((N-2)*(np.sum(ex))))
    errc=np.sqrt(np.sum(ei2)*s_xx/(N*(N-2)*np.sum(ex)))
    #covarience calculation
    cov=-meanx*(errm**2)
    return m,c,errm,errc,cov

cl=3*(10**8)
#importing the csv file as a data frame
df=pd.read_csv('data.csv')
#storing the necessary data
ri=df['NASA dist'] #stores distance of galaxies from earth
u_ri=df['sigma dist'] #stores uncertainty in distance
zi=df['NASA z'] 
vi=[] #stores velocity of recession of galaxies, evaluated from redshift 
for i in range (0,len(zi)):
    vi.append(zi[i])
u_vi=[]
u_zi=df['sigma z'] #stores uncertainty in redshift and in turn in velocity
for i in range (0,len(u_zi)):
    u_vi.append(u_zi[i])
    
    
m0,c0,sig_m,sig_c,rho=linreg(vi,ri)
print (m0,c0,sig_m,sig_c)


m_range=np.arange(3000,4500,0.001)
c_range=np.arange(2.5,5.5,0.01)


x1=[]
y1=[]
x2=[]
y2=[]

I=np.ones(len(m_range))
print (len(I))
print (len(m_range))
chisq_min=chisq(vi,ri,m0,c0,1)
l1=chisq_min+2.3
l2=chisq_min+6.17
print (chisq_min)


for c in c_range:
    chisq_val=chisq(vi,ri,m_range,c,I)
    for i in range (0,len(chisq_val)):
        term=chisq_val[i]
        if (np.abs(term-l1)<=0.1):
            x1.append(c)
            y1.append(m_range[i])
            print (term, c, m_range[i])
        elif (np.abs(term-l2)<=0.1):
            x2.append(c)
            y2.append(m_range[i])
            print (term, c, m_range[i])
   
plt.scatter(x1,y1,label="68% contour")
plt.scatter(x2,y2,label="95% contour")
plt.scatter(c0,m0)
plt.xlabel("intercept c")
plt.ylabel("slope m")
plt.grid()
plt.legend()