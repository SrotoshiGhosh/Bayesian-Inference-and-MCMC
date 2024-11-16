# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:49:05 2024
Implementation of Weighted Least Squares Fit and plotitng of the Chi-Squared contours
@author: SROTOSHI GHOSH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def wlsf(x,y,sig_y):
    S_0=np.sum(1/(sig_y**2))
    S_x=np.sum(x/(sig_y**2))
    S_y=np.sum(y/(sig_y**2))
    S_xx=np.sum((x**2)/(sig_y**2))
    S_yy=np.sum((y**2)/(sig_y**2))
    S_xy=np.sum((x*y)/(sig_y**2))
    m=((S_0*S_xy)-(S_x*S_y))/((S_0*S_xx)-((S_x)**2))
    c=((S_xy*S_x)-(S_y*S_xx))/((S_x**2)-(S_xx*S_0))
    sigc=np.sqrt(S_xx/(S_0*S_xx - S_x**2))
    sigm=np.sqrt(S_0/(S_0*S_xx - S_x**2))
    return m,c,sigm,sigc

def chisq(x,y,m,c,I,erry):
    s=0
    for i in range (0,len(x)):
        term=(y[i]*I-m*x[i]-c*I)/erry[i]
        s+=(term**2)
    return s

df=pd.read_csv('data.csv')
r=df['NASA dist']
u_r=df['sigma dist']
v=df['NASA z']

m_best,c_best,err_m,err_c=wlsf(v,r,u_r)
print ("best fit slope : ",m_best)
print("error : ",err_m)
print ("best fit intercept",c_best)
print ("error : ",err_c)

y_fit=[]
for i in range (0,len(v)):
    y_fit.append(m_best*v[i]+c_best)    
plt.plot(v,y_fit,label="WLS-Fitted line")
plt.plot(v,r,'r.',label="Data Points")
plt.xlabel("Redshift z")
plt.ylabel("Distance in Mpc")
plt.legend()
plt.grid()
chisq_min=chisq(v,r,m_best,c_best,1,u_r)
print ("Minimum value of chi-square",chisq_min)

#in order to plot the contours 
m_range=np.arange(4000,5000,0.001)
c_range=np.arange(-0.7,0.7,0.01)


x1=[]
y1=[]
x2=[]
y2=[]

I=np.ones(len(m_range))

l1=chisq_min+2.3
l2=chisq_min+6.17


for c in c_range:
    chisq_val=chisq(v,r,m_range,c,I,u_r)
    for i in range (0,len(chisq_val)):
        term=chisq_val[i]
        if (np.abs(term-l1)<=0.1):
            x1.append(c)
            y1.append(m_range[i])
            print (term-chisq_min, c, m_range[i])
        elif (np.abs(term-l2)<=0.1):
            x2.append(c)
            y2.append(m_range[i])
            print (term-chisq_min, c, m_range[i])
   
plt.scatter(x1,y1,label="68% contour")
plt.scatter(x2,y2,label="95% contour")
plt.scatter(c_best,m_best,label="true value of slope and intercept")
plt.xlim((-0.8,0.8))
plt.ylim((4400, 5050))
plt.xlabel("intercept c")
plt.ylabel("slope m")
plt.grid()
plt.legend()


np.savetxt('data1.txt', np.column_stack((x1, y1)), header='c m')
np.savetxt('data2.txt', np.column_stack((x2, y2)), header='c m')
