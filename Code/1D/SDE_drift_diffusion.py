import numpy as np
import matplotlib.pyplot as plt

# 1D forward stochastic process
# dXt  = mu*dt + sigma*dWt
# X(0) = X0 (const.)

# Numerical setup
T = 2  # terminal time
M = 100  # number of iterations
dt = T / M  # time step size
N = 2000  # number of particles

# Parameters in f and g
mu = 3
sigma = 0.5

# Initial condition
X_0 = 0

# Exact mean and std at T
mu_ex = X_0 + mu * T
std_ex = sigma * np.sqrt(T)

# SDE setup
f = lambda x, t: mu
g = lambda x, t: sigma

# Euler-Maruyama method
Xh_0 = np.zeros((N, M + 1))
Xh_0[:, 0] = X_0

for i in range(M):
    ti = i * dt
    Xh_0[:, i + 1] = Xh_0[:, i] + f(Xh_0[:, i], ti) * dt + g(Xh_0[:, i], ti) * np.sqrt(dt) * np.random.randn(N)

# Compute mean and std from discrete data
mu_sde = np.mean(Xh_0[:, M])
std_sde = np.std(Xh_0[:, M])

# Plotting
plt.figure(figsize=(12, 5))

# Plot time evolution
plt.subplot(1, 2, 1)
time_grid = np.linspace(0, T, M+1)
plt.plot(time_grid, Xh_0.T)
plt.ylim([-10, 10])
plt.xlabel('time')
plt.ylabel('X_t')
plt.title('Forward stochastic process')

# Plot distribution at t=T
plt.subplot(1, 2, 2)
x_values = np.linspace(-10, 10, 200)
plt.plot(-0.1 * np.ones(N), Xh_0[:, M], 'k.', alpha=0.1)
#plt.plot(x_values, (1 / (std_ex * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mu_ex) / std_ex)**2), lw=2)
plt.ylim([-10, 10])
plt.axis('off')

# Display exact and numerical values for mean and standard deviation
print(f'exact.mean = {mu_ex:.6f}')
print(f'numer.mean = {mu_sde:.6f}')
print('---------------------')
print(f'exact.std  = {std_ex:.6f}')
print(f'numer.std  = {std_sde:.6f}')

plt.tight_layout()
plt.show()
