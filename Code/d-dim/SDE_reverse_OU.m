clc; clear all; close;
%% d-dim reverse Ornstein-Uhenbeck process
%  dXt  = -beta*Xt*dt + sigma*dWt
%  X(0) = X0 ~ N(mu_0, sigma_0^2)
%  ------------------------------ 
%  mean(t) = mu_0*exp(-beta*t)
%  Cov(t)  = sigma_0^2*exp(-2*beta*t) + sigma^2/(2*beta)*( 1 - exp(-2*beta*t) )
%  score(x,t) = -(x-mean(t))/Cov(t)
%  ------------------------------ 
%% numerical setup
d  = 3;    % problem dimension
T  = 2;    % terminal time
M  = 1000; % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... parameters in f and g ...
beta  = 1;
sigma = 5;
% ... exact mean and std ...
mu_0    = randn(d,1);
sigma_0 = 2;
% ... initial condition ...
mu  = @(t) mu_0*exp(-beta*t);
std = @(t) sqrt( exp(-2*beta*t)*sigma_0^2 + sigma^2/(2*beta)*(1-exp(-2*beta*t)) );
Xh_0 = mvnrnd(mu(T),std(T).^2*eye(d),N)'; 
%% Ornstein-Uhenbeck process
f = @(x,t) -beta*x;
g = @(x,t) sigma;
s = @(x,t) -( x - mu(t) )./std(t).^2;
%% Euler-Maruyama method (backward)
for i = M:-1:1
   ti = i*dt;
   Xh_0 = Xh_0 + ( f(Xh_0,ti) - g(Xh_0,ti).^2.*s(Xh_0,ti) )*(-dt) ...
        + g(Xh_0,ti)*sqrt(dt)*randn(d,N);
end
%% Compute mean and cov from discrete data
mu_sde  = sum(Xh_0,2)/N;
cov_sde = cov(Xh_0')*(1-1/N);
%% Output
disp('exact.mean = '); disp(' ');
disp(mu_0');
disp('numer.mean = '); disp(' ');
disp(mu_sde');
disp('---------------------');
disp('exact.Cov  = '); disp(' ');
disp(sigma_0^2*eye(d));
disp('numer.Cov  = '); disp(' ');
disp(cov_sde);