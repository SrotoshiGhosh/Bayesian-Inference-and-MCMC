# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:41:05 2024

confidence interval and t-test

@author: SROTOSHI GHOSH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

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
    return m,c,errm,errc

def t_pdf(t,n):
    x=1+(t**2)/n
    y=np.power(x,-1/2*(n+1))
    y=y*(math.gamma((n+1)/2)/(np.sqrt(n*np.pi)*math.gamma(n/2)))
    return y

def t_int(t,n):
    a=-10000
    b=t
    N=10**6
    h=(b-a)/N
    s=t_pdf(a,n)+t_pdf(b,n)
    for i in range (1,N):
        x=a+i*h
        if (i%2==0):
            s+=2*t_pdf(x,n)
        else:
            s+=4*t_pdf(x,n)
    s=s*(h/3.0)
    return s

def func(t, alpha, nu):
    target = 1 - alpha / 2
    return t_int(t, nu) - target

def func_derivative(t, nu):
    return t_pdf(t, nu)

def newton_raphson(alpha, nu, initial_guess=1.0, tol=1e-6, max_iter=100):
    t = initial_guess
    for i in range(max_iter):
        f_val = func(t, alpha, nu)
        f_deriv = func_derivative(t, nu)
        if abs(f_val) < tol:
            return t
        t -= f_val / f_deriv
    raise RuntimeError("Newton-Raphson method did not converge")
    

cl=3*(10**8)
#importing the csv file as a data frame
df=pd.read_csv('data.csv')
#storing the necessary data
ri=df['NASA dist'] #stores uncertainty in distance
zi=df['NASA z']
vi=[] #stores velocity of recession of galaxies, evaluated from redshift 
for i in range (0,len(zi)):
    vi.append(zi[i]*cl)
'''
u_vi=[]
u_zi=df['sigma z'] #stores uncertainty in redshift and in turn in velocity
for i in range (0,len(u_zi)):
    u_vi.append(cl*u_zi)
''' 
    
m0,c0,sig_m,sig_c=linreg(vi,ri)
print ("Best fit slope with uncertainty : ",m0," +/- ",sig_m)
print ("Best fit intercept with uncertainty : ",c0," +/- ",sig_c)
#fitting and plotting
yfit=[]
for x in vi:
    t=m0*x+c0
    yfit.append(t)
plt.plot(vi,yfit,label="fitted line")
plt.plot(vi,ri,'.',color="black",label="observed data")
plt.grid()
plt.xlabel("velocity in m/s")
plt.ylabel("distance in Mpc")

#calculating 95% confidence interval of slope and intercept
ci_m=newton_raphson(0.32,len(vi)-2)*sig_m
print ("68% confidence interval of slope : ",m0," +/- ",ci_m)
ci_c=newton_raphson(0.32,len(vi)-2)*sig_c
print ("68% confidence interval of intercept : ",c0," +/- ",ci_c)

#SE(y)
ey2=0
ex2=0
xmean=np.mean(vi)
for i in range (0,len(ri)):
  term1=(ri[i]-yfit[i])**2
  term2=(vi[i]-xmean)**2
  ex2+=term2
  ey2+=term1
ey2=np.sqrt(ey2/len(vi)-2)



#confidence interval of regression line itself
y_upperci=[]
y_lowerci=[]
for i in range (0,len(yfit)):
  k=1/len(yfit) + (vi[i]-xmean)**2/(ex2)
  k=np.sqrt(k)
  k=k*ey2
  u=yfit[i]+(newton_raphson(0.32,len(vi)-2)*k)
  print (u)
  y_upperci.append(u)
  l=yfit[i]-(newton_raphson(0.32,len(vi)-2)*k)
  print (l)
  y_lowerci.append(l)

plt.plot(vi,y_upperci,"--",color="red",label="confidence interval")
plt.plot(vi,y_lowerci,"--",color="red")
plt.legend()