# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:56:46 2024

Verification of the Central Limit Theorem for Cauchy Distribution

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt


#this function evaluates the Gaussian distribution formula
def gaussian(x,m,s):
    t=(x-m)**2/(2*s*s)
    P=np.exp(-1*t)/(s*np.sqrt(2*np.pi))
    return P


n1=10**5 #no. of experiments
n2=int(input("Enter the number of trials in each experiment : ")) #no.of sums

a=0 #mean of distribution (location factor)
b=1 #standard deviation of distrbution (scale factor)
data=[]
s=0
c=0
for i in range (1,n1):
    x=np.random.standard_cauchy(n2)
    #generates n2 random cauchy variables
    for j in x:
        if (-5<=j<=5): 
            #as cauchy distribution has heavy tails and it needs to be adjusted accordingly 
            s=np.sum(x)
            c+=1
    s=s/c #mean value of each sum is evaluated 
    data.append(s)
    
#evaluating ideal number of bins from the Sturges' Rule
n_bins=int(1+np.ceil(np.log2(len(data))))

plt.hist(data, bins=n_bins, density=True, color="blue", alpha=0.4,) 
#we plot the histogram corresponding to the density of each outcome on y-axis and the outcome 
#itself on the x-axis
#density controls the height of histograms according to the number of 
#trials by dividing the actual height by number of trials  

#to plot the corresponding normal distribution curve and prove the central limit theorem
m=0
#mean of distribution derived from central limit theorem and analytical calculations
sd=1/np.sqrt(n2)
#standard deviation derived from central limit theorem and analytical calculations

x=np.arange(-5,5,0.01)
PDF=[]
for i in x:
    PDF.append(gaussian(i,m,sd))

plt.plot(x,PDF, color="black", label="gaussian distribution curve")

plt.legend()
plt.grid()
title = "Verification of central limit theorem for Cauchy Distribution with \n"
title += f"trials = {n2}"
plt.title(title)

