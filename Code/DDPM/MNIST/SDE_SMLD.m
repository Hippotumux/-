clc; clear all; close;
%% 1D forward Langevin Dynamics process
%  dXt  = sqrt( d( sigma(t)^2 )/dt )*dWt
%  X(0) = X0 (const.)
%  ------------------------------ 
%  sigma = sigma_min*( sigma_max/sigma_min )^(t/T)
%  mean = X0
%  var  = sigma_min^2*( ( sigma_max/sigma_min )^(2*t/T) - 1 )
%  ------------------------------ 
%% numerical setup
d  = 784;  % problem dimension
T  = 2;    % terminal time
M  = 100;  % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... parameters in f and g ...
sigma_min = 0.2;
sigma_max = 2;
% ... initial condition ...
% X_0   = rand(d,1);
load X_0.mat;
Xh_0  = zeros(d,N) + X_0;
% ... exact mean and std at t = T ...
mu_ex  = X_0;
cov_ex = (sigma_max^2 - sigma_min^2)*eye(d);
%% SDE setup
f = @(x,t) 0;
g = @(x,t) sigma_min*( sigma_max/sigma_min )^(t/T)*sqrt( 2/T*log(sigma_max/sigma_min) );
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