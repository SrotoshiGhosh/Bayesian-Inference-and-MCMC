# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:37:54 2024

Drawing the contours on the m-c plane

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cl=3*(10**8)
#importing the csv file as a data frame
df=pd.read_csv('data.csv')
#storing the necessary data
ri=df['NASA dist'] #stores distance of galaxies from earth
u_ri=df['sigma dist'] #stores uncertainty in distance
zi=df['NASA z'] 
vi=[] #stores velocity of recession of galaxies, evaluated from redshift 
for i in range (0,len(zi)):
    vi.append(zi[i])
u_vi=[]
u_zi=df['sigma z'] #stores uncertainty in redshift and in turn in velocity
for i in range (0,len(u_zi)):
    u_vi.append(u_zi[i])


#chi-squared fitting
S_x=np.sum(vi/(u_ri**2))
S_y=np.sum(ri/(u_ri**2))
S_xx=np.sum((vi/u_ri)**2)
S_yy=np.sum((ri/u_ri)**2)
S_xy=np.sum((ri*vi)/(u_ri**2))
S_0=np.sum(1/(u_ri**2))


sig_m=1/np.sqrt(S_xx)
sig_c=1/np.sqrt(S_0)
rho=S_x/np.sqrt(S_xx*S_0)
m_best=((S_0*S_xy)-(S_x*S_y))/((S_0*S_xx)-((S_x)**2))
c_best=((S_y*S_xx)-(S_x*S_xy))/((S_0*S_xx)-((S_x)**2))
chisq_min=S_yy+((S_0*(S_xy**2))-(2*S_x*S_y*S_xy)+(S_xx*(S_y**2)))/(((S_x)**2)-(S_0*S_xx))

print (m_best, c_best)
'''
#chisquared fitting
yfit=[]
for i in vi:
    yfit.append(m_best*i+c_best)
    
plt.plot(vi,yfit)
plt.scatter(vi,ri)
'''

#defining grids to plot contour
delm=np.sqrt(((sig_m*3)**2)/(1-rho)**2)
delc=np.sqrt(((sig_c*3)**2)/(1-rho)**2)
m_range=np.linspace(m_best-delm,m_best+delm,100)
c_range=np.linspace(c_best-delc,c_best+delc,100)
m,c=np.meshgrid(m_range,c_range)

#evaluating chi_sq term
chisq=(m-m_best)**2/(sig_m**2)+(c-c_best)**2/(sig_c**2)+(2*rho*(m-m_best)*(c-c_best))/(sig_m*sig_c)+chisq_min

#plotting the 68% contour
a=chisq_min+2.3
b=chisq_min+6.17
p=chisq_min
print (p)
Z=plt.contour(c, m, chisq, levels=[a,b], colors=['blue','red'])
plt.plot(c_best,m_best,'.',color='black')

plt.xlabel('intercept')
plt.ylabel('slope')
plt.title('Chi-square 68% and 95% contours in A-B plane')
plt.axvline(x=(c_best-sig_c), color='b', linestyle='dotted', label='1 sigma line')
plt.axvline(x=(c_best+sig_c), color='b', linestyle='dotted', label='1 sigma line')

plt.axvline(x=(c_best-2*sig_c), color='g', linestyle='dotted', label='2 sigma line')
plt.axvline(x=(c_best+2*sig_c), color='g', linestyle='dotted', label='2 sigma line')


plt.axhline(y=(m_best-sig_m), color='b', linestyle='dotted', label='1 sigma line')
plt.axhline(y=(m_best+sig_m), color='b', linestyle='dotted', label='1 sigma line')
plt.axhline(y=(m_best-2*sig_m), color='g', linestyle='dotted', label='2 sigma line')
plt.axhline(y=(m_best+2*sig_m), color='g', linestyle='dotted', label='2 sigma line')
