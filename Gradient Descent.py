# -*- coding: utf-8 -*-
"""

Gradient Descent Method for linear regression 

@author: SROTOSHI GHOSH
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def sq_error(x, y, theta):
    m, c = theta
    s = np.sum((y - m * x - c) ** 2)
    return s

def grad(x, y, theta):
    m, c = theta
    grad_m = -2 * np.sum(x * (y - m * x - c))
    grad_c = -2 * np.sum(y - m * x - c)
    return np.array([grad_m, grad_c])

def grad_desc(x, y, theta, step, tol):
    iter_count = 0
    while iter_count < 10000:
        value = sq_error(x, y, theta)
        grad_value = grad(x, y, theta)
        theta = theta - grad_value * step
        new_value = sq_error(x, y, theta)
        if np.abs(new_value - value) <= tol:
            break
        iter_count += 1
    return theta

# Loading the data and extracting the columns
df = pd.read_csv('data.csv')
r = df['NASA dist']
z = df['NASA z']

# Normalizing the data to ensure convergence of gradient descent method
r_mean, r_std = np.mean(r), np.std(r)
z_mean, z_std = np.mean(z), np.std(z)
r_normalized = (r - r_mean) / r_std
z_normalized = (z - z_mean) / z_std

# Initializing random seed and theta(parameters)
random.seed(0)
theta = [random.random(), random.random()]

# Setting step size and tolerance
step = 0.01
tolerance = 1e-6

# Running gradient descent
m, c = grad_desc(z_normalized, r_normalized, theta, step, tolerance)

# Converting the normalized parameters back to the original scale
m_original = m * (r_std / z_std)
c_original = c * r_std + r_mean - m * z_mean * (r_std / z_std)

# Printing the result
print(f"m: {m_original}, c: {c_original}")

# Generating the predicted y-values using original scale parameters
y_pred = m_original * z + c_original

# Plotting the results
plt.plot(z, y_pred, label='Fitted Line')
plt.scatter(z, r, color='red', label='Data Points')
plt.xlabel('velocity in units of c')
plt.ylabel('distance in Mpc')
plt.legend()
plt.show()
