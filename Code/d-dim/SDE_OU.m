clc; clear all; close;
%% d-dim forward stochastic process
%  dXt  = F*Xt*dt + G*dWt
%  X(0) = X0 ~ N(mu_0,SIGMA_0)
%  ----------------------------- 
%  mean = exp(t*F)*mu_0
%  Cov  = solved by RK4
%  -----------------------------
%% numerical setup
d  = 3;    % problem dimension
T  = 2;    % terminal time
M  = 100;  % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... construct F and G ...
F = randn(d,d);
F = -F*F';
G = randn(d,d);
G = G*G';
% ... initial condition ...
mu_0    = randn(d,1);
SIGMA_0 = randn(d,d);
SIGMA_0 = SIGMA_0'*SIGMA_0;
Xh_0 = mvnrnd(mu_0,SIGMA_0,N)';
% ... exact mean and cov at t = T ...
mu_ex  = expm(T*F)*mu_0;
cov_ex = RK4_OU_covariance(T,F,G,SIGMA_0);
%% SDE setup
f = @(x,t) F*x;
g = @(x,t) G;
%% Euler-Maruyama method
for i = 1:M
   ti = (i-1)*dt;
   Xh_0 = Xh_0 + f(Xh_0,ti)*dt + g(Xh_0,ti)*sqrt(dt)*randn(d,N);
end
%% Compute mean and cov from discrete data
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