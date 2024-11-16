# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:08:41 2024

Verification of the Central Limit Theorem for Normal Distribution

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt

#this function evaluates the gaussian distribution formula
def gaussian(x,m,s):
    t=((x-m)**2)/(2*s*s)
    P=np.exp(-1*t)/(s*np.sqrt(2*np.pi))
    return P

n1=10**5 #no. of throws of dice 
n2=int(input("Enter the number of dice to be thrown at the same time : ")) #no.of dice

data=[]
s=0
for i in range (1,n1):
    x=np.random.randint(1,7,n2) 
    #generates random numbers between 1 to 6, simulating dice throws, n2 signifies 
    #the number of random numbers to be generated at a time
    s=np.sum(x)
    s=s/n2 #mean value of each throw is evaluated 
    data.append(s)

#evaluating ideal number of bins from the Sturges Rule
n_bins=int(1+np.ceil(np.log2(len(data))))

plt.hist(data, bins=n_bins, density=True, color="blue", edgecolor="black", alpha=0.4,) 
#we plot the histogram corresponding to the density of each outcome on y-axis and the outcome 
#itself on the x-axis
#density controls the height of histograms according to the number of 
#throws by dividing the actual height by number of throws  

#to plot the corresponding normal distribution curve and prove the central limit theorem
m=3.5 
#mean of distribution derived from central limit theorem and analytical calculations
sd=1.70783/np.sqrt(n2) 
#standard deviation derived from central limit theorem and analytical calculations
xmin,xmax=plt.xlim()
x=np.arange(xmin,xmax,0.01)
PDF=[]
for i in x:
    PDF.append(gaussian(i,m,sd))

plt.plot(x,PDF, color="black", label="gaussian distribution curve")
plt.xlabel("mean value of throws of dice")
plt.ylabel("frequency of outcomes")
plt.legend()
plt.grid()
title = "Verification of central limit theorem for \n"
title += f"no. of dice thrown at a time = {n2}"
plt.title(title)