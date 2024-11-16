# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 23:09:22 2024

Bayesian Parameter Estimation for Non-linear Model 

@author: SROTOSHI GHOSH
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc

#defining the desired model to be fitted
def model(H0,W0,z):
    H= H0 * np.sqrt((W0*(1+z)**3) + 1 - W0)
    return H

def chisq(H,W,xi,yi,ui):
    k=np.sum(((yi-model(H,W,xi))/ui)**2)
    return k

#defining the likelihood distribution
def likelihood(xi,yi,sigi,H_p,W_p):
    ym = model(H_p,W_p,xi)
    return -0.5 * np.sum(((yi - ym) / sigi)**2)

#defining the prior distributions
def prior(H_p, W_p, Hmean, Hsd, Wmean):
    prior_H = -0.5 * ((H_p - Hmean) / Hsd)**2
    prior_W = np.log(Wmean)
    return prior_H + prior_W

#defining the posterior distribution
def posterior(H_p,W_p,Hmean,Hsd,Wmean,xi,yi,sigi):
    return likelihood(xi,yi,sigi,H_p,W_p) + prior(H_p,W_p,Hmean,Hsd, Wmean)

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
proposal_width_H = 2 # Adjusted proposal width
proposal_width_W = 0.03  # Adjusted proposal width
Hmean, Hsd = 73.8, 2.4  # Prior mean and standard deviation for Hubble parameter
Wmean = 0.42 # Prior mean and standard deviation for Density parameter

# Initialize the chain with informed values
samples = np.zeros((num_samples, 2))
samples[0, :] = [Hmean, Wmean]

num_accepted=0
# Metropolis-Hastings sampling
for i in range(1, num_samples):
    currentH, currentW = samples[i-1, :]
    
    # Propose new slope and intercept
    proposedH = np.random.normal(currentH, proposal_width_H)
    proposedW = np.random.normal(currentW, proposal_width_W)
    
    # Calculate acceptance probability
    posterior_current = posterior(currentH,currentW,Hmean,Hsd,Wmean,xi,yi,sigi)
    posterior_proposed = posterior(proposedH,proposedW,Hmean,Hsd,Wmean,xi,yi,sigi)
    acceptance_prob = min(1, np.exp(posterior_proposed - posterior_current))
    
    # Accept or reject the proposed parameters
    if acceptance_prob > np.random.rand():
        samples[i, :] = [proposedH, proposedW]
        num_accepted+=1
    else:
        samples[i, :] = [currentH, currentW]

# Analyze the results
burn_in = 10000
Hsamples = samples[burn_in::10, 0]
Wsamples = samples[burn_in::10, 1]

acceptance_rate = num_accepted/(num_samples-1)
print("Acceptance rate:", acceptance_rate)


H_mean = np.mean(Hsamples)
H_sd=np.std(Hsamples)
W_mean = np.mean(Wsamples)
W_sd=np.std(Wsamples)

print("Estimated H: ",H_mean)
print("Estimated W:",W_mean)
print("uncertainty in H: ",H_sd)
print("uncertainty in W:",W_sd)

#Plotting the chisq contours
chisq_min=chisq(H_mean,W_mean,xi,yi,sigi)
l1=chisq_min+2.3
l2=chisq_min+6.17

H_range=np.linspace(H_mean-7*H_sd,H_mean+10*H_sd,1000)
W_range=np.linspace(W_mean-7*W_sd,W_mean+10*W_sd,1000)
H,W=np.meshgrid(H_range,W_range)
chisq_values = np.zeros((len(W_range), len(H_range)))
for i in range(len(W_range)):
    for j in range(len(H_range)):
        chisq_values[i, j] = chisq(H_range[j],W_range[i],xi,yi,sigi)

# Create the figure and define the grid layout
fig = plt.figure(figsize=(10, 8))

# Top-left plot (1st figure in the L-shape)
ax1 = plt.subplot2grid((2, 2), (1, 1))
pdf1 = sc.stats.norm.pdf(H_range, H_mean, H_sd)
ax1.hist(Hsamples, bins=50, density=True, alpha=0.4, color='g')
ax1.plot(H_range, pdf1, color='green')
ax1.axvline(x=H_mean, color='red', linestyle='--', label='Mean of distribution')
ax1.set_title("Marginal Posterior for H_0", fontsize=12)
ax1.set_xlabel("H_0", fontsize=10)
ax1.set_ylabel("Density", fontsize=10)
ax1.legend(fontsize=8)

# Bottom-left plot (2nd figure in the L-shape)
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax2.plot(Hsamples, Wsamples, 'o', markersize=2, alpha=0.3)
ax2.contour(H, W, chisq_values, levels=[l1, l2], colors=['black', 'black'])
ax2.plot(H_mean, W_mean, '*', color='black', label="Mean of H_0 and Omega_m0")
ax2.set_title("Joint Posterior", fontsize=12)
ax2.set_xlabel("H_0", fontsize=10)
ax2.set_ylabel("Omega_m0", fontsize=10)
ax2.legend(fontsize=8)

# Bottom-right plot (3rd figure in the L-shape)
ax3 = plt.subplot2grid((2, 2), (0, 0))
pdf2 = sc.stats.norm.pdf(W_range, W_mean, W_sd)
ax3.hist(Wsamples, bins=50, density=True, alpha=0.4, color='b')
ax3.plot(W_range, pdf2, color='b')
ax3.axvline(x=W_mean, color='red', linestyle='--', label='Mean of distribution')
ax3.set_title("Marginal Posterior for Omega_m0", fontsize=12)
ax3.set_xlabel("Omega_m0", fontsize=10)
ax3.set_ylabel("Density", fontsize=10)
ax3.legend(fontsize=8)

# Adjust the layout to avoid overlap
plt.tight_layout()

# Show the plot
plt.show()
'''
# Plot the results
plt.figure(figsize=(8, 6))
plt.ylim(0.2,0.425)
plt.contour(H,W , chisq_values, levels=[l1, l2])
plt.plot(H_mean, W_mean, '*', color='black')
plt.xlabel('H')
plt.ylabel('W')
plt.title('1sigma and 2sigma contours')
plt.grid(True)
plt.show()

# Plot the results
plt.figure()
plt.plot(Hsamples, Wsamples, 'o', markersize=2, alpha=0.5)
plt.contour(H,W , chisq_values, levels=[l1, l2], colors=['black','black'])
plt.plot(H_mean, W_mean, '*', color='black')
plt.xlabel('H')
plt.ylabel('W')
plt.title('Metropolis-Hastings Sampling of Joint Posterior of H and W')
plt.grid(True)
plt.show()


# Plot the marginal distribution of slope
pdf1=sc.stats.norm.pdf(H_range,H_mean,H_sd)
plt.figure(figsize=(8, 6))
plt.hist(Hsamples, bins=50, density=True, alpha=0.6, color='g')
plt.plot(H_range,pdf1,color='black')
plt.xlabel('H')
plt.ylabel('Density')
plt.title('Marginal Distribution of H')
plt.grid(True)
plt.show()

# Plot the marginal distribution of intercept
pdf2=sc.stats.norm.pdf(W_range,W_mean,W_sd)
plt.figure(figsize=(8, 6))
plt.hist(Wsamples, bins=50, density=True, alpha=0.6, color='b')
plt.plot(W_range,pdf2, color='black')
plt.xlabel('W')
plt.ylabel('Density')
plt.title('Marginal Distribution of W')
plt.grid(True)
plt.show()

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(Hsamples,color='g')
plt.xlabel('Iteration Number')
plt.ylabel('Sample Values')
plt.title('Trace Plot for H Samples')
plt.grid(True)
plt.show()

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(Wsamples)
plt.xlabel('Iteration Number')
plt.ylabel('Sample Values')
plt.title('Trace Plot for W Samples')
plt.grid(True)
plt.show()
'''