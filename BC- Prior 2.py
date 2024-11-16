# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 17:46:09 2024
Obtains posterior distribution for parallax problem for a specific prior choice 
@author: SROTOSHI GHOSH
"""

import numpy as np
import matplotlib.pyplot as plt

def L(w,r,s): #defines the gaussian likelihood
    t=np.exp((-1*(w-1/r)**2)/(2*s**2))
    P=t/(s*np.sqrt(2*np.pi))
    return P

def normalization(w,r,sd):
    x=[]
    for i in r:
        if (i>=0):
            x.append(i)
    a=np.min(x) 
    #lower limit of integration taken to be the smallest positive number as defined prior is 0 for -ve distances
    b=np.max(x)
    n=10**3
    h=(b-a)/n
    s=L(w,a,sd)+L(w,b,sd)
    for i in range (1,int(n)):
        t=a+(i*h)
        if (i%2==0):
            s+=2*L(w,t,sd)
        else:
            s+=4*L(w,t,sd)
    s=s*(h/3.0)
    return s

f=[0.25] #fractional parallax uncertainty
w=-1/100 #measured parallax
sd=[]
for i in range (0,len(f)):
    sd.append(f[i]*w)

r=np.arange(-10,1000,0.001) #a range of distances are defined

p=[] 
rlim=1000
#defining the necessary prior which in this case was one terminated at a certain limit which
#is the largest distance we expect in our survey of stars
for i in r:
    if (rlim>=i>0):
        p.append(1)
    else:
        p.append(0)


for k in range (0,len(sd)):
    post_pdf=[] #evaluation of the normalized posterior pdf
    N=normalization(w,r,sd[k])#in order to evaulate the normalization constant
    l=f"f = {f[k]} "
    for i in range (0,len(r)):
        term=(p[i]*L(w,r[i],sd[k]))/N
        post_pdf.append(term)
    plt.plot(r,post_pdf,label=l)
    
plt.xlabel("r")
plt.ylabel("P(r|w)")
plt.legend()
