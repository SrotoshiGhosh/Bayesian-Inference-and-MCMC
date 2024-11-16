# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:43:12 2024

Linear Regression using Ordinary Least Squares Fitting 

@author: SROTOSHI GHOSH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def lin_reg(xi,yi):
    N=len(xi)
    s_xi=np.sum(xi)
    s_yi=np.sum(yi)
    xi2=[]
    xiyi=[]
    for i in range (0,N):
        xi2.append(xi[i]**2)
        xiyi.append(xi[i]*yi[i])
    s_xi2=np.sum(xi2)
    s_xiyi=np.sum(xiyi)
    #slope (m)
    m=((N*s_xiyi)-(s_xi*s_yi))/((N*s_xi2)-(s_xi**2))
    #intercept (c)
    c=(s_yi-(m*s_xi))/N
    #calculation of errors
    ei=[]
    ei2=[]
    for i in range (0,N):
        term=yi[i]-m*xi[i]-c
        ei.append(term)
        ei2.append(term**2)
    meanxi=np.sum(xi)/N
    exi=[]
    for i in range (0,N):
        term=(xi[i]-meanxi)**2
        exi.append(term)
    er_slope=np.sqrt(np.sum(ei2)/((N-2)*(np.sum(exi))))
    er_inter=np.sqrt(np.sum(ei2)*s_xi2/(N*(N-2)*np.sum(exi)))
    #covarience calculation
    cov=-meanxi*(er_slope**2)
    return m,er_slope,c,er_inter,xi2,xiyi,ei,ei2,exi,cov

cl=3*(10**8)
#importing the csv file as a data frame
df=pd.read_csv('data.csv')
#storing the necessary data
ri=df['NASA dist'] #stores distance of galaxies from earth
u_ri=df['sigma dist'] #stores uncertainty in distance
zi=df['NASA z'] 
vi=[] #stores velocity of recession of galaxies, evaluated from redshift 
for i in range (0,len(zi)):
    vi.append(zi[i]*cl)
print (vi)
u_zi=df['sigma z'] #stores uncertainty in redshift and in turn in velocity
u_vi=[]
for i in range (0,len(u_zi)):
    u_vi.append(cl*u_zi[i])
    
m,errm,c,errc,xi2,xiyi,ei,ei2,exi,cov=lin_reg(vi,ri)

yfit=[]
for i in range (0,len(vi)):
    yfit.append((m*vi[i])+c)

plt.plot(vi,ri,"ro",label="observed data")
#plt.errorbar(vi,ri,u_ri,u_vi,fmt='.',capsize=2,label="observed data with error bars")
plt.ylabel("distance in Mpc")
plt.xlabel("velocity in m/s")
plt.grid()
plt.legend()

print (" The slope of the fitted line is : ",m," with an uncertainty of : ",errm)
print (" The intercept of the fitted line is : ",c," with an uncertainty of : ",errc)  

#evaluation of fractional errors 
N=len(ri)
fx=[]
fy=[]
for i in range (0,N):
    fy.append(u_ri[i]/ri[i])
    fx.append(u_vi[i]/vi[i])
mfx=np.sum(fx)/N
mfy=np.sum(fy)/N

print (" The mean fractional error in velocity is : ",mfx)
print (" The mean fractional error in distance is : ",mfy)

#creating the covariance matrix
k=2 #as 2 parameters are being estimated
#the matrix is of k*k square dimensions
CovM=np.zeros((k,k))
for i in range (0,k):
    for j in range (0,k):
        if (i!=j):
            CovM[i,j]=cov
        else:
            if (i==0):
                CovM[i,j]=errm**2
            if (i==1):
                CovM[i,j]=errc**2

print (" The covariance matrix is given as : ")
print (CovM)
       
print ("Comparing the covariance matrix with in built linear regression : ")
res=linregress(ri,vi)
meanxi=np.sum(ri)/len(ri)
mlr=res.slope #slope fit from linear regression function
slope_var=res.stderr**2 
intercept_var=slope_var * np.mean(xi2)

# Compute covariance matrix manually
covariance_matrix = np.array([[slope_var, -meanxi * slope_var],
                              [-meanxi * slope_var, intercept_var]])
print(covariance_matrix)

#creating data frame to store as csv
data={'xi':vi, 'sigma xi':u_vi, 'frac err in x':fx, 'yi':ri, 'sigma yi':u_ri, 'frac err in y':fy, 
      'xi^2':xi2, 'xi*yi':xiyi, 'yi(fitted)':yfit, 'residue ei':ei, 'ei^2':ei2}

df2=pd.DataFrame(data)
df2.to_csv('Lin Reg Data2.csv')
