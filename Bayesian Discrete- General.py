# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 8 11:34:07 2024
GENERAL CODE FOR FINDING OUT POSTERIOR PROBABILITIES OF A DISCRETE DISTRIBUTION, APPLIED TO BREAST CANCER PROBLEM
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
pdm=np.zeros(100,float)
print("Enter the array of likelihoods(do end with a 0 when you are done with all inputs) : ")
n_pdm=0
for i in range (0,len(pdm)):
    pdm[i]=input()
    if (pdm[i]==0):
        n_pdm=i
        break
p_dm=slicing(pdm,n_pdm)


#evaluation of marginal likelihood or evidence or normalization constant P(D)
n=len(p_m)
p_d=0
for i in range (0,n):
    p_d+=p_m[i]*p_dm[i]
    
    
#evaluation of posterior P(M|D)
p_md=np.zeros(n)
for i in range (0,n):
    p_md[i]=(p_m[i]*p_dm[i])/p_d
print(" The posterior distribution : ", p_md)


