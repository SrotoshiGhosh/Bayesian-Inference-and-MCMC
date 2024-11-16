# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 23:30:33 2024

@author: Srotoshi Ghosh

This program intends to determine the posterior predictive distribution through MCMC techniques for a
non-linear flat CDM model of the universe, given data for the expansion rate and the redshift.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc

#defining the desired model to be fitted
def model(H0,W0,z):
    H= H0 * np.sqrt((W0*(1+z)**3) + 1 - W0)
    return H

#defining the likelihood distribution
def likelihood(xi,yi,sigi,H_p,W_p):
    ym = model(H_p,W_p,xi)
    return -0.5 * np.sum(((yi - ym) / sigi)**2)

#defining the prior distributions
def prior(H_p, W_p, Hmean, Hsd, W_min, W_max):
    # Gaussian prior for H_p
    prior_H = -0.5 * ((H_p - Hmean) / Hsd) ** 2
    # Flat prior for W_p in the range [W_min, W_max]
    if W_min <= W_p <= W_max:
        prior_W = 0  # Log(1) = 0 in log space, so flat
    else:
        prior_W = -np.inf  # Outside the range, the prior probability is 0
    return prior_H + prior_W

#defining the posterior 
def posterior(H_p, W_p, Hmean, Hsd, W_min, W_max, xi, yi, sigi):
    return likelihood(xi, yi, sigi, H_p, W_p) + prior(H_p, W_p, Hmean, Hsd, W_min, W_max)


df = pd.read_csv('ratradata.csv')
xi = df['z'].values  # Distance of galaxies from Earth
yi = df['H(z)'].values
sigi = df['sigma H'].values
xi = np.array(xi)
yi = np.array(yi)
sigi = np.array(sigi)
N = len(xi)

# Metropolis-Hastings sampling parameters
num_samples = 1000000
proposal_width_H = 1 # Adjusted proposal width
proposal_width_W = 0.05  # Adjusted proposal width
Hmean, Hsd = 68, 2.8  # Prior mean and standard deviation for Hubble parameter
W_min, W_max= 0.2,0.4 # Prior minimum and maximum values

# Initialize the chain with informed values
samples = np.zeros((num_samples, 2))
samples[0, :] = [Hmean, 0.28]
num_accepted=0
# Metropolis-Hastings sampling
for i in range(1, num_samples):
    currentH, currentW = samples[i-1, :]
    
    # Propose new slope and intercept
    proposedH = np.random.normal(currentH, proposal_width_H)
    proposedW = np.random.normal(currentW, proposal_width_W)
    
    # Calculate acceptance probability
    posterior_current = posterior(currentH,currentW,Hmean,Hsd,W_min,W_max,xi,yi,sigi)
    posterior_proposed = posterior(proposedH,proposedW,Hmean,Hsd,W_min,W_max,xi,yi,sigi)
    acceptance_prob = min(1, np.exp(posterior_proposed - posterior_current))
    
    # Accept or reject the proposed parameters
    if acceptance_prob > np.random.rand():
        samples[i, :] = [proposedH, proposedW]
        num_accepted+=1
    else:
        samples[i, :] = [currentH, currentW]

# Analyze the results
burn_in = 100000
Hsamples = samples[burn_in::10, 0]
Wsamples = samples[burn_in::10, 1]

acceptance_rate = num_accepted/(num_samples-1)
print("Acceptance rate:", acceptance_rate)
H_mean = np.mean(Hsamples)
W_mean = np.mean(Wsamples)
print("Estimated H: ",H_mean)
print("Estimated W:",W_mean)

#calculating predictive posterior distribution
zp= 0.3800
posterior_pred_samples=[]
for i in range (0,len(Hsamples)):
    H0=Hsamples[i]
    W0=Wsamples[i]
    Hp0=model(H0,W0,zp)
    posterior_pred_samples.append(Hp0)
    
Hp_mean=np.mean(posterior_pred_samples)
Hp_sd=np.std(posterior_pred_samples)

print ("Posterior predictive mean for z_p (H_p) : ",Hp_mean)
print ('Posterior predictive std : ',Hp_sd)
x=np.linspace(Hp_mean-4*Hp_sd,Hp_mean+4*Hp_sd,1000)
pdf=sc.stats.norm.pdf(x,Hp_mean,Hp_sd)


# Plot the posterior predictive distribution at yp
plt.figure(figsize=(8, 6))
plt.hist(posterior_pred_samples, bins=50, density=True, alpha=0.4, color='purple')
plt.plot(x,pdf,color='purple')
plt.axvline(x=Hp_mean,color='black', linestyle='--', label="Estimated Value")
plt.axvline(x=81.5,color='red', linestyle='-', label="True Value")
plt.xlabel('Posterior predictive values')
plt.ylabel('Density')
plt.title(f'Posterior Predictive Distribution at z_p={zp}')
plt.grid(True)
plt.legend()
plt.show()

'''
plt.figure(figsize=(8, 6))
plt.plot(posterior_pred_samples)
plt.xlabel('Iteration Number')
plt.ylabel('Sample Value')
plt.title('Trace Plot of Distribution Samples')
plt.grid(True)
plt.show()
'''