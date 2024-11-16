# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:46:46 2024

Generalized example for Gibbs Sampling

@author: SROTOSHI GHOSH
    
"""

import numpy as np
import matplotlib.pyplot as plt

# Mean vector and covariance matrix
mean = np.array([0, 0])
cov = np.array([[1, 0.9], [0.9, 1]])

# Number of samples
num_samples = 10000

# Initialize the chain
samples = np.zeros((num_samples, 2))
samples[0, :] = np.random.normal(size=2)

# Gibbs sampling
for i in range(1, num_samples):
    # Sample x given y
    x_mean = mean[0] + cov[0, 1] / cov[1, 1] * (samples[i-1, 1] - mean[1])
    x_var = cov[0, 0] - cov[0, 1]**2 / cov[1, 1]
    samples[i, 0] = np.random.normal(x_mean, np.sqrt(x_var))
    
    # Sample y given x
    y_mean = mean[1] + cov[1, 0] / cov[0, 0] * (samples[i, 0] - mean[0])
    y_var = cov[1, 1] - cov[1, 0]**2 / cov[0, 0]
    samples[i, 1] = np.random.normal(y_mean, np.sqrt(y_var))

# Plot the samples
plt.plot(samples[:, 0], samples[:, 1], 'o', markersize=2, alpha=0.5)
plt.xlabel('theta 1')
plt.ylabel('theta 2')
plt.title('Gibbs Sampling of Bivariate Normal Distribution')
plt.grid(True)
plt.show()

plt.plot(samples[:,0])
plt.xlabel("Number of Iterations")
plt.ylabel("Samples")
plt.title("Trace Plot of Theta 1")
'''
plt.plot(samples[:,1])
plt.xlabel("Number of Iterations")
plt.ylabel("Samples")
plt.title("Trace Plot of Theta 2")
'''

