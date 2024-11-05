# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:40:06 2024

Monte Carlo Integration-generalized

@author: DELL
"""
import numpy as np


#sampling distribution, which is simpler
def q(x):
    m=0
    s=1
    y=np.exp(-((x-m)**2)/(2*s**2))/(s*np.sqrt(2*np.pi))
    return y

#target function
def f(x):
    m=0
    s=0.01
    y=np.exp(-((x-m)**2)/(2*s**2))/(s*np.sqrt(2*np.pi))
    return y


N=10
while N<(10**6):
    #importance sampling
    x1=np.random.normal(0,1,N)
    pi=f(x1)
    qi=q(x1)
    wi=pi/qi
    I_is=np.mean(wi)
    print ("Value of integration estimated using importance sampling for N = ",N,"is : ",I_is)
    diff_is=np.abs(1-I_is)
    print ("Difference : ",diff_is)
    #monte-carlo sampling
    a=-100
    b=+100
    x2=np.random.uniform(a,b,N)
    y2=f(x2)
    I_mc=np.mean(y2)*(b-a)
    print ("Value of integration estimated using Monte Carlo uniform sampling for N = ",N,"is : ",I_mc)
    diff_mc=np.abs(1-I_mc)
    print ("Difference : ",diff_mc)
    print("")
    N=N*10

'''
plt.plot(x1,pi,'g.',label='target fn')
plt.plot(x1,qi,'r.',label='sampling fn')
plt.legend()
'''