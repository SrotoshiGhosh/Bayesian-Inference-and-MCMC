# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:46:39 2024

General Example for implementation of the Metropolis-Hastings Algorithm

@author: SROTOSHI GHOSH
"""

import numpy as np
import matplotlib.pyplot as plt

#target distribution
def g(x):
    m1=3
    m2=-1
    s1=0.5
    s2=0.7
    t1=np.exp(-0.5 * ((x - m1) / s1)**2) / (s1 * np.sqrt(2 * np.pi))
    t2=np.exp(-0.5 * ((x - m2) / s2)**2) / (s2 * np.sqrt(2 * np.pi))
    return 0.5*(t1+t2)

#returns random sample from porposal distribution 
#the proposal distribution is a Gaussian
#it has mean 1 and standard deviation 0
def Q(x):
    return np.random.normal(x,1.0)

def metropolishastings(x_ini):
    N=10**5
    samples=np.zeros(N)
    samples[0]=x_ini
    
    for i in range (0,N):
        x_prop=x_ini + Q(0)#getting a random sample from the proposal distribution
        ratio=min(1,g(x_prop)/g(x_ini))#the factor is cancelled as Q is chosen to be symmetric
        r=np.random.uniform(0,1)
        if(r<=ratio):
            x_ini=x_prop
        
        samples[i]=x_ini
    
    return samples

x=-1 
s=metropolishastings(x)

plt.hist(s,bins=30,density=True,alpha=0.5,edgecolor="black",label="MCMC Samples")

'''
xi=np.linspace(-10,10,10000)
yi=g(xi)
plt.plot(xi,yi,label="target distribution")
plt.xlabel("Values of x")
plt.ylabel("Density of g(x)")
plt.title("Metropolis-Hastings Algorithm Simulation")
plt.legend()
'''          
plt.plot(s)
plt.xlabel("Iteration Number")
plt.ylabel("Sample Value")
plt.title("Trace Plot of Samples")

        

