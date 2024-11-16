# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:27:28 2024

Bayesian Linear Regression using Gibbs Sampling Algorithm

@author: SROTOSHI GHOSH
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import my_linreg as mlr

#Loading the data for regression
cl=3*10**8
df=pd.read_csv('data.csv')
ri=df['NASA dist'].values #stores distance of galaxies from earth
zi=df['NASA z'].values
vi=zi*cl
u_ri=df['sigma dist'].values
ri=np.array(ri)
vi=np.array(vi)
u_ri=np.array(u_ri)
N=len(vi)
slope,intercept,er_slope,er_inter,cov_val=mlr.wlsf(vi,ri,u_ri)
#the weighted least squares fit is used to determine the contours

#to plot the contour lines
chisq_min=mlr.chisq(vi,ri,slope,intercept,u_ri)
l1=chisq_min+2.3
l2=chisq_min+6.17

slope_range=np.linspace(slope-4*er_slope,slope+4*er_slope,1000)
intercept_range=np.linspace(intercept-4*er_inter,intercept+4*er_inter,1000)
m,c=np.meshgrid(slope_range,intercept_range)
chisq_values = np.zeros((len(intercept_range), len(slope_range)))
for i in range(len(intercept_range)):
    for j in range(len(slope_range)):
        chisq_values[i, j] = mlr.chisq(vi, ri, slope_range[j], intercept_range[i], u_ri)

# Mean vector and covariance matrix
# The mean matrix represents the ordinary least square fitted value of slope and intercept
mean = np.array([slope,intercept])
# The covariance matrix is obtained from ordinary least squares fitting as well
cov = np.array([[er_slope**2,cov_val],[cov_val,er_inter**2]])
print (cov)
# Number of samples
num_samples = 10000

# Initializing the chain with any random samples
samples = np.zeros((num_samples, 2))
samples[0, :] = np.random.multivariate_normal(mean, cov)

# Gibbs sampling
for i in range(1, num_samples):
    # Sampling slope given intercept
    x_mean = mean[0] + cov[0, 1] / cov[1, 1] * (samples[i-1, 1] - mean[1])
    x_var = cov[0, 0] - cov[0, 1]**2 / cov[1, 1]
    samples[i, 0] = np.random.normal(x_mean, np.sqrt(x_var))
    
    # Sampling intercept given slope
    y_mean = mean[1] + cov[1, 0] / cov[0, 0] * (samples[i, 0] - mean[0])
    y_var = cov[1, 1] - cov[1, 0]**2 / cov[0, 0]
    samples[i, 1] = np.random.normal(y_mean, np.sqrt(y_var))
    
slopem=np.mean(samples[:,0])
slopesd=np.std(samples[:,0])
interm=np.mean(samples[:,1])
intersd=np.std(samples[:,1])

print (slopem, slopesd)
print (interm, intersd)

# Create the figure and define the grid layout
fig = plt.figure(figsize=(10, 8))

# Top-left plot (1st figure in the L-shape)
ax1 = plt.subplot2grid((2, 2), (1, 1))
ax1.hist(samples[:, 0], bins=50, density=True, alpha=0.4, color='g')
ax1.axvline(x=slopem, color='red', linestyle='-', label='Distribution mean')
ax1.axvline(x=slope, color='black', linestyle='--', label='Classical estimate')
ax1.set_title("Slope Marginal Posterior", fontsize=12)
ax1.set_xlabel("Slope", fontsize=10)
ax1.set_ylabel("Density", fontsize=10)
ax1.legend(fontsize=8)

# Bottom-left plot (2nd figure in the L-shape)
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax2.plot(samples[:, 0], samples[:, 1], 'o', markersize=2, alpha=0.3)
ax2.contour(m, c, chisq_values, levels=[l1, l2], colors=['black', 'black'])
ax2.plot(slope, intercept, '*', color='black', label="Classical estimate")
ax2.plot(slopem, interm, '*', color='red', label="Distribution mean")
ax2.set_title("Joint Posterior", fontsize=12)
ax2.set_xlabel("Slope", fontsize=10)
ax2.set_ylabel("Intercept", fontsize=10)
ax2.legend(fontsize=8)

# Bottom-right plot (3rd figure in the L-shape)
ax3 = plt.subplot2grid((2, 2), (0, 0))
ax3.hist(samples[:, 1], bins=50, density=True, alpha=0.4, color='b')
ax3.axvline(x=interm, color='red', linestyle='-', label='Distribution mean')
ax3.axvline(x=intercept, color='black', linestyle='--', label='Classical estimate')
ax3.set_title("Intercept Marginal Posterior", fontsize=12)
ax3.set_xlabel("Intercept", fontsize=10)
ax3.set_ylabel("Density", fontsize=10)
ax3.legend(fontsize=8)

# Adjust the layout to avoid overlap
plt.tight_layout()

# Show the plot
plt.show()


'''
# Ploting the joint distribution of samples
plt.figure(figsize=(8, 6))
plt.plot(samples[:, 0], samples[:, 1], 'o', markersize=2, alpha=0.5)
plt.contour(m, c, chisq_values, levels=[l1, l2], colors=['black', 'red'])
plt.plot(slope, intercept, '*', color='black')
plt.xlabel('slope')
plt.ylabel('intercept')
plt.title('Gibbs Sampling of Joint Posterior of Slope and Intercept')
plt.grid(True)
plt.show()

# Plot the marginal distribution of X
plt.figure(figsize=(8, 6))
plt.hist(samples[:, 0], bins=50, density=True, alpha=0.6, color='g')

plt.xlabel('slope')
plt.ylabel('Density')
plt.title('Marginal Distribution of Slope')
plt.grid(True)
plt.show()

# Plot the marginal distribution of Y
plt.figure(figsize=(8, 6))
plt.hist(samples[:, 1], bins=50, density=True, alpha=0.6, color='b')
plt.xlabel('intercept')
plt.ylabel('Density')
plt.title('Marginal Distribution of Intercept')
plt.grid(True)
plt.show()

# Plot the marginal distribution of X
plt.figure(figsize=(8, 6))
plt.plot(samples[:, 0], color='g')
plt.xlabel('Iteration Number')
plt.ylabel('Sample Values')
plt.title('Trace Plot of Slope Samples')
plt.grid(True)
plt.show()

# Plot the marginal distribution of Y
plt.figure(figsize=(8, 6))
plt.plot(samples[:, 1], color='b')
plt.xlabel('Iteration Number')
plt.ylabel('Sample Values')
plt.title('Trace Plot of Intercept Samples')
plt.grid(True)
plt.show()
'''


