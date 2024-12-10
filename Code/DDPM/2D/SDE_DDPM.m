clc; clear all; close;
%% 2D forward DDPM process
%  X_k = sqrt( 1-beta_k )*X_{k-1} + sqrt( beta_k )*Z
%  X(0) = X0 (const.)
%  ------------------------------ 
%  mean = sqrt( bar{alpha}_k*X0 )
%  var  = 1 - bar{alpha}_k
%  ------------------------------ 
%% numerical setup
T  = 2;    % terminal time
M  = 100;  % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... parameters in f and g ...
beta      = (1:M)*dt^2;
alpha     = 1 - beta;
alpha_bar = zeros(1,M); alpha_bar(1) = alpha(1);
for i = 2:M
alpha_bar(i) = alpha(i)*alpha_bar(i-1);
end
% ... initial condition ...
X_0  = [5;6];
Xh_0 = zeros(2,N) + X_0;
% ... exact mean and std at t = T ...
mu_ex  = sqrt(alpha_bar(end))*X_0;
cov_ex = (1 - alpha_bar(end))*eye(2);
%% Euler-Maruyama method
for i = 1:M    
   ti = (i-1)*dt; 
   Xh_0 = sqrt( 1 - beta(i) )*Xh_0 + sqrt( beta(i) )*randn(2,N);
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