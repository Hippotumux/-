clc; clear all; close;
%% 2D reverse Ornstein-Uhenbeck process
%  dXt  = 
%  X(0) = X0 (const.)
%  ------------------------------ 
%  mean  = mu_0*exp(-beta*t)
%  var   = sigma_0^2*exp(-2*beta*t) + sigma^2/(2*beta)*( 1 - exp(-2*beta*t) )
%  ------------------------------ 
%% numerical setup
T  = 10;   % terminal time
M  = 1000; % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... parameters in f and g ...
beta  = @(t) t;
sigma = @(t) sqrt(beta(t));
% ... exact mean and std ...
X_0 = [2;5];
% ... initial condition ...
mu  = @(t) exp(-t^2/4)*X_0;
std = @(t) sqrt(1 - exp(-t^2/2));
Xh_0 = mvnrnd(mu(T),std(T).^2*eye(2),N)'; 
%% Ornstein-Uhenbeck process
f = @(x,t) -beta(t)/2*x;
g = @(x,t) sigma(t);
s = @(x,t) -( x - mu(t) )./std(t).^2;
eps = @(x,t) -std(t)*s(x,t);
%% Euler-Maruyama method (backward)
for i = M:-1:1
   ti = i*dt;
%    Xh_0 = Xh_0 + ( f(Xh_0,ti) - g(Xh_0,ti).^2.*s(Xh_0,ti) )*(-dt) ...
%         + g(Xh_0,ti)*sqrt(dt)*randn(2,N);

%    Xh_0 = ( 1 + 1/2*beta(ti)*dt )*Xh_0 - beta(ti)*dt/std(ti)*eps(Xh_0,ti) + ...
%         + sqrt( beta(ti)*dt )*randn(2,N);

   Xh_0 = 1/sqrt( 1 - beta(ti)*dt )*Xh_0 - beta(ti)*dt/std(ti)*eps(Xh_0,ti) + ...
        + sqrt( beta(ti)*dt )*randn(2,N);
end
%% Compute mean and std from discrete data
mu_sde  = sum(Xh_0,2)/N;
cov_sde = cov(Xh_0')*(1-1/N);
%% Output;
disp('exact.mean = '); disp(' ');
disp(X_0');
disp('numer.mean = '); disp(' ');
disp(mu_sde');
disp('---------------------');
disp('exact.Cov  = '); disp(' ');
disp(0*eye(2));
disp('numer.Cov  = '); disp(' ');
disp(cov_sde);