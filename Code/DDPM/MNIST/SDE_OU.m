clc; clear all; close;
%% 2D forward Ornstein-Uhenbeck process
%  dXt  = -beta(t)/2*Xt*dt + sqrt(beta(t))*dWt
%  X(0) = X0 (const.)
%  ------------------------------ 
%  mean = exp(-T^2/4)*X0
%  var  = 1 - exp(-T^2/2)
%  ------------------------------ 
%% numerical setup
d  = 784;  % problem dimension
T  = 2;    % terminal time
M  = 100;  % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... parameters in f and g ...
beta  = @(t) t;
sigma = @(t) sqrt(beta(t));
% ... initial condition ...
% X_0  = randn(d,1);
load X_0.mat;
Xh_0 = zeros(d,N) + X_0;
% ... exact mean and std at t = T ...
mu_ex  = exp(-T^2/4)*X_0;
cov_ex = (1 - exp(-T^2/2))*eye(d);
%% SDE setup
f = @(x,t) -beta(t)/2*x;
g = @(x,t) sigma(t);
%% Euler-Maruyama method
for i = 1:M
%    Xh_0(:, i+1) = Xh_0(:, i) - beta*Xh_0(:, i)*dt + sigma*sqrt(dt)*randn(N,1);    
   ti = (i-1)*dt; 
   Xh_0 = Xh_0 + f(Xh_0,ti)*dt + g(Xh_0,ti)*sqrt(dt)*randn(d,N);
end
%% Compute mean and std from discrete data
mu_sde  = sum(Xh_0,2)/N;
cov_sde = cov(Xh_0')*(1-1/N);
%% Output
disp('exact.mean = '); disp(' ');
disp(mu_ex');
disp('numer.mean = '); disp(' ');
disp(mu_sde');
disp('---------------------');
disp('exact.Cov  = '); disp(' ');
disp(cov_ex);
disp('numer.Cov  = '); disp(' ');
disp(cov_sde);