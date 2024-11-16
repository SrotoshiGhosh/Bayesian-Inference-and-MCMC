# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:48:41 2024

Bayesian Coin Toss- Gaussian Prior

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
from scipy.integrate import simps
from scipy.signal import find_peaks

def normalize(x,y):
    area=simps(y,x) 
    #utilizing built-in simpson integration to evaluate the area under curve and normalize
    return y/area

#simulating tossing of coins
n_sim=100000 #number of experiments 
N=int(input(" Enter the number of the coins being tossed at a time : "))
'''
M=0 
for j in range (0,N):
    k=np.random.randint(0,2,1)
    #assuming 0 to represent tails and 1 to represent heads
    if (k==1):
        M+=1
frac=M/N #fraction of heads
print (" The number of heads obtained when ", N, " coins are tossed : ", M)
print (" The probability of getting heads : ", '%4.4f'%frac)
'''
#defining the prior which follows a beta distribution where we assume mean and sd to be as follows
m=0.5
sd=0.1
x=np.linspace(10**(-5),1-10**(-5),1000)
prior=sc.norm.pdf(x,m,sd)

#defining the likelihood which follows a binomial distribution
M=8
x=np.linspace(10**(-5),1-10**(-5),1000)
likelihood=sc.binom.pmf(M,N,x)
peak, _ = find_peaks(likelihood)
xmax=x[peak]
ymax=likelihood[peak]
print( "The maximum probability of getting heads from likelihood : ",xmax[0])

#evaluating the posterior pdf
post=[]
for i in range (0,len(x)):
    post.append((prior[i]*likelihood[i]))
post=normalize(x,post) 
   
plt.plot(x,prior,color="red", label="prior")
plt.plot(x,likelihood,color="blue", label="likelihood")
plt.plot(x,post,color="green", label="posterior probability")
plt.grid()
plt.legend()
plt.xlabel(f"probability of getting {M} heads")
plt.ylabel("posterior probability")
title = f"no. of coins tossed = {N} and no. of heads obtained = {M}"
plt.title(title)
