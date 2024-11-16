# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:20:37 2024

Implements various goodness of fit tests for linear regression

@author: SROTOSHI GHOSH
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def r(x,y,xy,x2,y2,N):
    term= (N*np.sum(xy)- np.sum(x)*np.sum(y))
    term2= np.abs((N*np.sum(x2)-np.sum(x)**2)*(N*np.sum(y2)-np.sum(y)**2))
    return term/np.sqrt(term2)

def R2(y,ei2):
    ydiff=[]
    m=np.sum(y)/len(y)
    for i in range (0,len(y)):
        ydiff.append((y[i]-m)**2)
    term=np.sum(ei2)/np.sum(ydiff)
    return 1-term


filepath1=r'C:\Users\DELL\Desktop\PYTHON PROGS\Internship\Lin Reg Data2.csv'
filepath2=r'C:\Users\DELL\Desktop\PYTHON PROGS\Internship\Lin Reg Data4.csv'
df1=pd.read_csv(filepath1)
ei1=df1['residue ei']
xi1=df1['xi']
df2=pd.read_csv(filepath2)
ei2=df2['residue ei']
xi2=df2['xi']

#plotting residues and looking for any conclusive pattern
#if residues are randomly distributed, then they are said to be of adequate fit

plt.plot(xi1,ei1,'.',label="y=mx+c model")
#plt.plot(xi2,ei2,'*',label="y=mx model")
plt.legend()
plt.grid()

#both residue distributions have a random arrangement about the axes
#hence, we cannot decide which is a better fit through this method

#coefficient of correlation(r)
N1=len(xi1)
yi1=df1['yi']
xiyi1=df1['xi*yi']
xi21=df1['xi^2']
yi21=[]
for i in range (0,N1):
    yi21.append(yi1[i]**2)
r1=r(xi1,yi1,xiyi1,xi21,yi21,N1)

N2=len(xi2)
yi2=df2['yi']
xiyi2=df2['xi*yi']
xi22=df2['xi^2']
yi22=[]
for i in range (0,N2):
    yi22.append(yi2[i]**2)
r2=r(xi2,yi2,xiyi2,xi22,yi22,N2)

print ("The coefficient of correlation for y=mx+c model : ",r1)
print ("The coefficient of correlation for y=mx model : ",r2)


#coefficient of determination(R^2)
ei21=df1['ei^2']
ei22=df2['ei^2']
det1=R2(yi1,ei21)
print ("The coefficient of determination of y=mx+c model : ",det1)
det2=R2(yi2,ei22)
print ("The coefficient of determination for the y=mx model : ",det2)

#evaluating the sum of squares of residues and sum of residues
s_ei1=np.sum(ei1)
s_ei2=np.sum(ei2)
s_ei21=np.sum(ei21)
s_ei22=np.sum(ei22)
print (" The sums of residues for y=mx+c and y=mx model are respectively : ",s_ei1,"and", s_ei2)
print (" The sum of squares of residues of y=mx+c and y=mx models respectively are : ",s_ei21, "and", s_ei22)

'''
the sum is less for y=mx+c in both cases, so that is chosen
'''

rse1= np.sqrt(s_ei21/(N2-2))
print (" The value of residual standard error for y=mx+c model is : ",rse1)
rse2= np.sqrt(s_ei22/(N2-1))
print (" The value of residual standard error for y=mx model is : ",rse2)

