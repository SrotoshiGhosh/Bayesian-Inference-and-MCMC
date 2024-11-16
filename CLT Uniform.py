# -*- coding: utf-8 -*-
"""
Created on Thu May 30 00:14:45 2024

@author: SROTOSHI GHOSH
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

a=-10 #lower limit
b=+10 #upper limit
data=[]
s=0
for i in range (1,n1):
    x=np.random.uniform(-10,10,n2)
    #generates n2 uniform variables at a time
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
m=(b+a)/2
#mean of distribution derived from central limit theorem and analytical calculations
sd=(b-a)/np.sqrt(12*n2) 
#standard deviation derived from central limit theorem and analytical calculations
xmin,xmax=plt.xlim()
x=np.arange(xmin,xmax,0.01)
PDF=[]
for i in x:
    PDF.append(gaussian(i,m,sd))

plt.plot(x,PDF, color="black", label="gaussian distribution curve")
plt.legend()
plt.grid()
plt.title("Central Limit Theorem Verification for Uniform Distribution")
