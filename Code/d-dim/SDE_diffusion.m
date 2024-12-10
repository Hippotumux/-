clc; clear all; close;
%% d-dim forward stochastic process
%  dXt  = SIGMA^(1/2)*dWt
%  X(0) = X0 (const.)
%  ------------------------- 
%  mean = X0
%  Cov  = t*SIGMA
%  ------------------------- 
%% numerical setup
d  = 3;    % problem dimension
T  = 2;    % terminal time
M  = 100;  % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... construct SIGMA and SIGMA^(1/2) ...
A       = randn(d,d);
[~,S,V] = svd(A);
SIGMA   = V*S.^2*V';
SIGMA_half = V*S*V';
% ... initial condition ...
X_0 = randn(d,1);
Xh_0 = zeros(d,N) + X_0;
% ... exact mean and cov at t = T ...
mu_ex  = X_0;
cov_ex = SIGMA*T;
%% SDE setup
f = @(x,t) 0;
g = @(x,t) SIGMA_half;
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
disp('L2 error = '); disp(' ');
disp(norm(mu_ex-mu_sde)/sqrt(d));
disp('---------------------');
disp('exact.Cov  = '); disp(' ');
disp(cov_ex);
disp('numer.Cov  = '); disp(' ');
disp(cov_sde);
disp('L2 error = '); disp(' ');
disp(norm(cov_ex(:)-cov_sde(:))/d);