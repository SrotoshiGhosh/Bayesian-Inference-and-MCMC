# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:21:09 2024

f-test

@author: SROTOSHI GHOSH
"""

import pandas as pd
import numpy as np
import math

def f(x,f1,f2):
    y=np.power(x,(f1/2)-1)
    y=y/np.power((1+(f1*x/f2)),((f1+f2)/2))
    return y

def fprob(F,d1,d2):
    a=F
    b=10**5
    n=10**6
    h=(b-a)/n
    s=f(a,d1,d2)+f(b,d1,d2)
    for i in range (1,n):
        t=a+i*h
        if (i%2==0):
            s+=2*f(t,d1,d2)
        else:
            s+=4*f(t,d1,d2)
    s=s*(h/3.0)
    s=s*(math.gamma((d1+d2)/2)*np.power(d1/d2,d1/2))/(math.gamma(d1/2)*math.gamma(d2/2))
    return s

#importing the required data generated from the linear regression of two models
#the model with lesser number of parameters chosen as the first model as prescribed by test
filepath2=r'C:\Users\DELL\Desktop\PYTHON PROGS\Internship\Lin Reg Data2.csv'
filepath1=r'C:\Users\DELL\Desktop\PYTHON PROGS\Internship\Lin Reg Data4.csv'
df1=pd.read_csv(filepath1)
df2=pd.read_csv(filepath2)
e2=df2['ei^2']
e1=df1['ei^2']

#value of degrees of freedom
d1=len(e1)-1
d2=len(e2)-2

#evaluation of F-statistic
F=(np.sum(e1)-np.sum(e2))*(d2)/(np.sum(e2)*(d1-d2))
PF=fprob(F,d1,d2)
print ("The value of F-statistic is : ",F)
print ("The probability of obtaining this particular F-statistic : ",PF)
if (PF>0.05):
    print ("This is greater than the 5% probability, and hence, the null hypothesis that the model with less parameters is a better fit, is true.")
else:
    print ("The model with more number of parameters fits the data better, null hypothesis is rejected.")