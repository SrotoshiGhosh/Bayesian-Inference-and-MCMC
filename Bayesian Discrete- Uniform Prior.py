# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 11:34:07 2024
GENERAL CODE FOR FINDING OUT POSTERIOR PROBABILITIES OF A DISCRETE DISTRIBUTION
@author: SROTOSHI GHOSH
"""

import numpy as np
import matplotlib.pyplot as plt

#This function alters the size of the originally assumed arrays of likelihood and prior
def slicing(arr,l):
    return arr[:l]

#taking input of priors P(M)
pm=np.zeros(100,float)
print("Enter the array of priors(do end with a 0 when you are done with all inputs) : ")
n_pm=0
for i in range (0,len(pm)):
    pm[i]=input()
    if (pm[i]==0):
        n_pm=i
        break
p_m=slicing(pm,n_pm)


#taking input of likelihood P(D|M)
n_pdm=n_pm #as the number of priors=number of likelihood
pdm=np.zeros(n_pdm,float)
print("Enter the array of likelihoods except the one which will vary uniformly : ")
for i in range (0,(n_pdm-1)):
    pdm[i]=input()
    if (pdm[i]==0):
        break
p_dm=[]
for i in range(0,(n_pdm-1)):
    p_dm.append(pdm[i])
    
k=int(input("Enter the index of the prior corresponding to which the uniform likelihood will be taken : "))
x=np.arange(0.01,1.0,0.01)

#evaluation of marginal likelihood or evidence or normalization constant P(D)
n=len(p_m)
#evaluation of posterior P(M|D)
pdf=[]
p_md=np.zeros(n)
for j in x:
    p_d=0
    p_dm.insert(k,j)#inserts each value of the uniform likelihood
    for i in range (0,n):
        p_d+=p_m[i]*p_dm[i] #marginal likelihood
    for i in range (0,n):
        p_md[i]=(p_m[i]*p_dm[i])/p_d #posterior distribution
    pdf.append(p_md[0])
    p_dm.remove(j)#removes the inserted value to operate it for the next iteration
    


plt.plot(x,pdf)
plt.xlabel("P(D|M'')")
plt.ylabel("P(M'|D)")
