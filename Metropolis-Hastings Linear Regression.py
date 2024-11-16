# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 17:30:57 2024
Bayesian Linear Regression, using Metropolis-Hastings Algorithm
@author: SROTOSHI GHOSH
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import my_linreg as mlr

# Load the data for regression
cl = 3 * 10**8
df = pd.read_csv('data.csv')
ri = df['NASA dist'].values  # Distance of galaxies from Earth
zi = df['NASA z'].values
vi = zi * cl
u_ri = df['sigma dist'].values
ri = np.array(ri)
vi = np.array(vi)
u_ri = np.array(u_ri)
N = len(vi)
slope,intercept,er_slope,er_inter,cov_val = mlr.wlsf(vi, ri, u_ri)

# Define the likelihood function
def likelihood(vi, ri, slope, intercept, sigma):
    model = slope * vi + intercept
    return -0.5 * np.sum(((ri - model) / sigma)**2)

# Define the prior distributions
def prior(slope, intercept, mu_slope, tau_slope, mu_intercept, tau_intercept):
    prior_slope = -0.5 * ((slope - mu_slope) / tau_slope)**2
    prior_intercept = -0.5 * ((intercept - mu_intercept) / tau_intercept)**2
    return prior_slope + prior_intercept

# Define the posterior distribution
def posterior(vi, ri, slope, intercept, sigma, mu_slope, tau_slope, mu_intercept, tau_intercept):
    return likelihood(vi, ri, slope, intercept, sigma) + prior(slope, intercept, mu_slope, tau_slope, mu_intercept, tau_intercept)

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
        
# Metropolis-Hastings sampling parameters
num_samples = 1000000
proposal_width_slope = 0.000001  # Adjusted proposal width
proposal_width_intercept = 1  # Adjusted proposal width
mu_slope, tau_slope = slope, er_slope  # Prior mean and standard deviation for slope
print(mu_slope, tau_slope)
mu_intercept, tau_intercept = intercept, er_inter  # Prior mean and standard deviation for intercept
print(mu_intercept,tau_intercept)

# Initialize the chain with informed values
samples = np.zeros((num_samples, 2))
samples[0, :] = [mu_slope, mu_intercept]

# Metropolis-Hastings sampling
for i in range(1, num_samples):
    current_slope, current_intercept = samples[i-1, :]
    
    # Propose new slope and intercept
    proposed_slope = np.random.normal(current_slope, proposal_width_slope)
    proposed_intercept = np.random.normal(current_intercept, proposal_width_intercept)
    
    # Calculate acceptance probability
    posterior_current = posterior(vi, ri, current_slope, current_intercept, u_ri, mu_slope, tau_slope, mu_intercept, tau_intercept)
    posterior_proposed = posterior(vi, ri, proposed_slope, proposed_intercept, u_ri, mu_slope, tau_slope, mu_intercept, tau_intercept)
    acceptance_prob = min(1, np.exp(posterior_proposed - posterior_current))
    
    # Accept or reject the proposed parameters
    if acceptance_prob > np.random.rand():
        samples[i, :] = [proposed_slope, proposed_intercept]
    else:
        samples[i, :] = [current_slope, current_intercept]

# Analyze the results
burn_in = 1000
slope_samples = samples[burn_in:, 0]
intercept_samples = samples[burn_in:, 1]

slope_mean = np.mean(slope_samples)
slope_sd=np.std(slope_samples)
intercept_mean = np.mean(intercept_samples)
inter_sd=np.std(intercept_samples)

print(f"Estimated slope: {slope_mean}")
print(f"Estimated intercept: {intercept_mean}")
print("uncertainty in slope: ",slope_sd)
print("uncertainty in intercept",inter_sd)

# Create the figure and define the grid layout
fig = plt.figure(figsize=(10, 8))

# Top-left plot (1st figure in the L-shape)
ax1 = plt.subplot2grid((2, 2), (1, 1))
ax1.hist(samples[:, 0], bins=50, density=True, alpha=0.4, color='g')
ax1.axvline(x=slope_mean, color='red', linestyle='-', label='Distribution Mean')
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
ax2.plot(slope_mean, intercept_mean, '*', color='red', label="Joint Posterior Mean")
ax2.set_title("Joint Posterior", fontsize=12)
ax2.set_xlabel("Slope", fontsize=10)
ax2.set_ylabel("Intercept", fontsize=10)
ax2.legend(fontsize=8)

# Bottom-right plot (3rd figure in the L-shape)
ax3 = plt.subplot2grid((2, 2), (0, 0))
ax3.hist(samples[:, 1], bins=50, density=True, alpha=0.4, color='b')
ax3.axvline(x=intercept_mean, color='red', linestyle='-', label='Distribution Mean')
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
# Plot the results
plt.figure(figsize=(8, 6))
plt.hist2d(slope_samples, intercept_samples, bins=50, density=True, cmap='Blues')
plt.colorbar(label='Density')
plt.xlabel('slope')
plt.ylabel('intercept')
plt.title('Metropolis-Hastings Sampling of Joint Posterior of Slope and Intercept')
plt.grid(True)
plt.show()

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(slope_samples, intercept_samples, 'o', markersize=2, alpha=0.5)
plt.xlabel('slope')
plt.ylabel('intercept')
plt.title('Metropolis-Hastings Sampling of Joint Posterior of Slope and Intercept')
plt.grid(True)
plt.show()

# Plot the marginal distribution of slope
plt.figure(figsize=(8, 6))
plt.hist(slope_samples, bins=50, density=True, alpha=0.6, color='g')
plt.xlabel('slope')
plt.ylabel('Density')
plt.title('Marginal Distribution of Slope')
plt.grid(True)
plt.show()

# Plot the marginal distribution of intercept
plt.figure(figsize=(8, 6))
plt.hist(intercept_samples, bins=50, density=True, alpha=0.6, color='b')
plt.xlabel('intercept')
plt.ylabel('Density')
plt.title('Marginal Distribution of Intercept')
plt.grid(True)
plt.show()

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(slope_samples)
plt.xlabel('Iteration Number')
plt.ylabel('Sample Values')
plt.title('Trace Plot for Slope Samples')
plt.grid(True)
plt.show()

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(intercept_samples)
plt.xlabel('Iteration Number')
plt.ylabel('Sample Values')
plt.title('Trace Plot for Intercept Samples')
plt.grid(True)
plt.show()
'''