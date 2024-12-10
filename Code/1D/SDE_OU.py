import numpy as np 
import matplotlib.pyplot as plt 
'''
 1D forward Ornstein-Uhenbeck process
%  dXt  = -beta*Xt*dt + sigma*dWt
%  X(0) = X0 (const.)
%  ------------------------------ 
%  mean = X0*exp(-beta*t)
%  var  = sigma^2/(2*beta)*( 1 - exp(-2*beta*t) )
%  ------------------------------ 
% numerical setup
'''

# numerical setup
T = 2 # terminal time 
M = 100  # number of iters
dt = T/M  # time step size 
N = 2000  # number of particles 

# parameters in f and g 
beta = 1
sigma = 0.5 

X_0 = 5 
mu_ex = X_0 * np.exp(-beta*T)
std_ex = np.sqrt( sigma^2/(2*beta)*(1-np.exp(-2*beta*T)) )

f = lambda x, t : -beta * x
g = lambda x, t : sigma 

Xh_0 = np.zeros((N, M+1))
Xh_0[:, 1] = X_0 

for i in range(M):
    ti = (i)*dt
