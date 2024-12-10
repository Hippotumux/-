clc; clear all; close;
%% 1D reverse Langevin Dynamics process
%  dXt  = 
%  X(0) = X0 (const.)
%  ------------------------------ 
%  sigma = sigma_min*( sigma_max/sigma_min )^(t/T)
%  mean = X0
%  var  = sigma_min^2*( ( sigma_max/sigma_min )^(2*t/T) - 1 )
%  ------------------------------ 
%% numerical setup
d  = 2;    % problem dimension
T  = 10;   % terminal time
M  = 1000; % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... parameters in f and g ...
sigma_min = 0.1;
sigma_max = 2;
% ... exact mean and std ...
X_0 = randn(d,1);
% ... initial condition ...
mu  = @(t) X_0;
std = @(t) sqrt( sigma_min^2*( ( sigma_max/sigma_min )^(2*t/T) - 1 ) );
Xh_0 = mvnrnd(mu(T),std(T).^2*eye(d),N)'; 
%% Ornstein-Uhenbeck process
f = @(x,t) 0;
g = @(x,t) sigma_min*( sigma_max/sigma_min )^(t/T)*sqrt( 2/T*log(sigma_max/sigma_min) );
s = @(x,t) -( x - mu(t) )./std(t).^2;
eps = @(x,t) -std(t)*s(x,t);
%% Euler-Maruyama method (backward)
for i = M:-1:1
   ti = i*dt;
   Xh_0 = Xh_0 + ( f(Xh_0,ti) - g(Xh_0,ti).^2.*s(Xh_0,ti) )*(-dt) ...
        + g(Xh_0,ti)*sqrt(dt)*randn(d,N);
end
%% Compute mean and std from discrete data
mu_sde  = sum(Xh_0,2)/N;
cov_sde = cov(Xh_0')*(1-1/N);
%% Output
disp('exact.mean = '); disp(' ');
disp(X_0');
disp('numer.mean = '); disp(' ');
disp(mu_sde');
disp('---------------------');
disp('exact.Cov  = '); disp(' ');
disp(0*eye(d));
disp('numer.Cov  = '); disp(' ');
disp(cov_sde);