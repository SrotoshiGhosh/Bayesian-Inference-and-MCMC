# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 19:29:57 2024

@author: SROTOSHI GHOSH
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def z(x,y):
    return np.abs((x-y)/y)

df=pd.read_csv('data.csv')

print(df)

#angular spread
a=df['Angular Spread']

#distance from earth in megaparsec Mpc
dist=[]
for i in range (0,len(a)):
    dist.append(22/a[i])
print (dist)

#wavelength of Ca-K line 
lk=df['L-K']
lk_ori=3933.7
c=3*10**8
#redshift calculation
z_lk=[]
for i in range (0,len(lk)):
    z_lk.append(z(lk[i],lk_ori))


#wavelength of Ca-H line
lh=df['L-H']
lh_ori=3968.5
#redshift calculation
z_lh=[]
for i in range (0,len(lh)):
    z_lh.append(z(lh[i],lh_ori))
    
#wavelength of H-alpha line
la=df['L-alpha']
la_ori=6562.8
#redshift calculation
z_la=[]
for i in range (0,len(la)):
    z_la.append(z(la[i],la_ori))
    
#average redshift
red=[]
for i in range (0,len(la)):
    term=(z_lk[i]+z_lh[i]+z_la[i])/3.0
    red.append(term)
    
#average velocity
vel=[]
for i in range (0,len(red)):
    vel.append(red[i]*c)
    
#fitting the data in a linear form
coeff=np.polyfit(dist,vel,1)
fit=np.poly1d(coeff)
x_fit=np.linspace(np.min(dist),np.max(dist),1000)
y_fit=fit(x_fit)
print (coeff)

plt.plot(x_fit,y_fit)
plt.plot(dist,vel,"*")
#creating dictionary of new columns
new={'Distance':dist,'Redshift of L-K':z_lk, 'Redshift of L-H':z_lh, 'Redshift of L-alpha':z_la, 
     'Average redshift':red, 'Average velocity':vel}

#adding the new columns to the DataFrame
for column_name, column_data in new.items():
    df[column_name] = column_data

#saving the updated DataFrame back to the same CSV file
df.to_csv('data.csv', index=False)

