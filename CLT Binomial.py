# -*- coding: utf-8 -*-
"""
Created on Wed May 29 23:19:59 2024

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt

#this function evaluates the Cauchy distribution formula
def gaussian(x,m,s):
    t=(x-m)**2/(2*s*s)
    P=np.exp(-1*t)/(s*np.sqrt(2*np.pi))
    return P

n1=10**5 #no. of experiments
n2=int(input("Enter the number of trials in each experiment : ")) #no.of sums

n=10 #no. of repeated trials
p=0.5#probability of success
data=[]
s=0
for i in range (1,n1):
    x=np.random.binomial(n,p,n2)
    #generates n2 random variables lying on the binomial distribution
    s=np.sum(x)
    s=s/n2 #mean value of each sum is evaluated 
    data.append(s)
    
#evaluating ideal number of bins from the Sturges' Rule
n_bins=int(1+np.ceil(np.log2(len(data))))

plt.hist(data, bins=n_bins, density=True, color="blue", edgecolor="black", alpha=0.4,) 
#we plot the histogram corresponding to the density of each outcome on y-axis and the outcome 
#itself on the x-axis
#density controls the height of histograms according to the number of 
#trials by dividing the actual height by number of trials 

#to plot the corresponding normal distribution curve and prove the central limit theorem
m=n*p
#mean of distribution derived from central limit theorem and analytical calculations
sd=np.sqrt(n*p*(1-p))/np.sqrt(n2) 
#standard deviation derived from central limit theorem and analytical calculations
xmin,xmax=plt.xlim()
x=np.arange(xmin,xmax,0.01)
PDF=[]
for i in x:
    PDF.append(gaussian(i,m,sd))

plt.plot(x,PDF, color="black", label="gaussian distribution curve")
plt.legend()
plt.grid()
title = "Verification of central limit theorem for Binomial Distribution with \n"
title += f"trials = {n2}"
plt.title(title)