clc; clear all; close;
%% d-dim reverse Ornstein-Uhenbeck process
%  dXt  = F*Xt*dt + G*dWt
%  X(0) = X0 ~ N(mu_0,SIGMA_0)
%  ------------------------------ 
%  mean(t) = exp(t*F)*mu_0
%  Cov(t)  = SIGMA(t) (solved by RK4)
%  score(x,t) = -inv(SIGMA(t))*( x - mean(t) )
%  ------------------------------ 
%% numerical setup
d  = 3;    % problem dimension
T  = 2;    % terminal time
M  = 1000; % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... construct F and G ...
F = randn(d,d);
F = -F*F';
G = randn(d,d);
G = G*G';
% ... exact mean and std at t = 0 ...
mu_0    = randn(d,1);
SIGMA_0 = randn(d,d);
SIGMA_0 = SIGMA_0'*SIGMA_0;
% ... initial condition ...
mu    = @(t) expm(t*F)*mu_0;
SIGMA = @(t) RK4_OU_covariance(t,F,G,SIGMA_0);
Xh_0  = mvnrnd(mu(T),SIGMA(T),N)'; 
%% Ornstein-Uhenbeck process
f = @(x,t) F*x;
g = @(x,t) G;
s = @(x,t) -SIGMA(t)\( x - mu(t) );
%% Euler-Maruyama method (backward)
for i = M:-1:1
   ti = i*dt;
   Xh_0 = Xh_0 + ( f(Xh_0,ti) - g(Xh_0,ti)*g(Xh_0,ti)'*s(Xh_0,ti) )*(-dt) ...
        + g(Xh_0,ti)*sqrt(dt)*randn(d,N);
   %% NOTE g*g' needs a careful treatment if G = G(Xt,t)! %%
   %% SIGMA(t) in score function can be computed sequentially %%
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
disp(SIGMA_0);
disp('numer.Cov  = '); disp(' ');
disp(cov_sde);