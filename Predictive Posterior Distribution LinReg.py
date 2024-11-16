# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:34:48 2024
Posterior Predictive Distribution to verify linear model 
@author: SROTOSHI GHOSH
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import my_linreg as mlr
import scipy.stats as sc

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
m, er_slope, c, er_inter, xi2, xiyi, ei, ei2, exi, cov_val = mlr.lin_reg(vi, ri)

# Define the likelihood function
def likelihood(ri, vi, slope, intercept, sigma):
    model = slope * vi + intercept
    return -0.5 * np.sum(((ri - model) / sigma)**2)

# Define the prior distributions
def prior(slope, intercept, mu_slope, tau_slope, mu_intercept, tau_intercept):
    prior_slope = -0.5 * ((slope - mu_slope) / tau_slope)**2
    prior_intercept = -0.5 * ((intercept - mu_intercept) / tau_intercept)**2
    return prior_slope + prior_intercept

# Define the posterior distribution
def posterior(ri, vi, slope, intercept, sigma, mu_slope, tau_slope, mu_intercept, tau_intercept):
    return likelihood(ri, vi, slope, intercept, sigma) + prior(slope, intercept, mu_slope, tau_slope, mu_intercept, tau_intercept)

# Metropolis-Hastings sampling parameters
num_samples = 100000
sigma = np.sqrt(np.sum(ei2)/(N-2))  # Use mean of uncertainties as an estimate for sigma
proposal_width_slope = 0.000001
proposal_width_intercept = 1.0
mu_slope, tau_slope = m, er_slope  # Prior mean and standard deviation for slope
mu_intercept, tau_intercept = c, er_inter  # Prior mean and standard deviation for intercept

# Initialize the chain with random values
samples = np.zeros((num_samples, 2))
samples[0, :] = [np.random.normal(mu_slope, tau_slope), np.random.normal(mu_intercept, tau_intercept)]

# Metropolis-Hastings sampling
acceptance_count = 0
for i in range(1, num_samples):
    current_slope, current_intercept = samples[i-1, :]
    
    # Propose new slope and intercept
    proposed_slope = np.random.normal(current_slope, proposal_width_slope)
    proposed_intercept = np.random.normal(current_intercept, proposal_width_intercept)
    
    # Calculate acceptance probability
    posterior_current = posterior(ri, vi, current_slope, current_intercept, sigma, mu_slope, tau_slope, mu_intercept, tau_intercept)
    posterior_proposed = posterior(ri, vi, proposed_slope, proposed_intercept, sigma, mu_slope, tau_slope, mu_intercept, tau_intercept)
    acceptance_prob = min(1, np.exp(posterior_proposed - posterior_current))
    
    # Accept or reject the proposed parameters
    if acceptance_prob > np.random.rand():
        samples[i, :] = [proposed_slope, proposed_intercept]
        acceptance_count += 1
    else:
        samples[i, :] = [current_slope, current_intercept]

# Analyze the results
burn_in = 1000
slope_samples = samples[burn_in:, 0]
intercept_samples = samples[burn_in:, 1]

slope_mean = np.mean(slope_samples)
intercept_mean = np.mean(intercept_samples)

print(f"Estimated slope: {slope_mean}")
print(f"Estimated intercept: {intercept_mean:.2f}")
print(f"Acceptance count: {acceptance_count}")

# Posterior predictive distribution at one particular value of x_p
xp = (0.028762*cl) # Example value for y_p #NGC 7603

# Computing the posterior predictive distribution at x_p
posterior_predictive_values = []

for j in range(len(slope_samples)):
    slope = slope_samples[j]
    intercept = intercept_samples[j]
    mean = slope * xp + intercept
    sampled_yp = np.random.normal(mean, sigma)
    posterior_predictive_values.append(sampled_yp)

m_pdf=np.mean(posterior_predictive_values)
sd_pdf=np.std(posterior_predictive_values)
print ("Posterior predictive mean for NGC 7603 (y_p) : ",m_pdf)
print ('Posterior predictive std for NGC 7603 (sigma y_p): ',sd_pdf)
x=np.linspace(m_pdf-4*sd_pdf,m_pdf+4*sd_pdf,1000)
pdf=sc.norm.pdf(x,m_pdf,sd_pdf)


# Plot the posterior predictive distribution at yp
plt.figure(figsize=(8, 6))
plt.hist(posterior_predictive_values, bins=50, density=True, alpha=0.4, color='purple')
plt.plot(x,pdf, color='purple')
plt.axvline(x=m_pdf, color='red', linestyle='-', label='Predicted y_p')
plt.axvline(x=121.74, color='black', linestyle='--', label='Observed y_p')
plt.xlabel('Posterior predictive values')
plt.ylabel('Density')
plt.title(f'Posterior Predictive Distribution at x_p={xp}')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(posterior_predictive_values)
plt.xlabel('Iteration Number')
plt.ylabel('Sample Value')
plt.title('Trace Plot of Distribution Samples')
plt.grid(True)
plt.show()



