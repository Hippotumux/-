clc; clear all; close;
%% 2D reverse DDPM process
%  X_k = sqrt( 1-beta_k )*X_{k-1} + sqrt( beta_k )*Z
%  X(0) = X0 (const.)
%  ------------------------------ 
%  mean = sqrt( bar{alpha}_k*X0 )
%  var  = 1 - bar{alpha}_k
%  ------------------------------ 
%% numerical setup
T  = 3;    % terminal time
M  = 1000; % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... parameters in f and g ...
beta      = (1:M)*dt^2;
alpha     = 1 - beta;
alpha_bar = zeros(1,M); alpha_bar(1) = alpha(1);
for i = 2:M
alpha_bar(i) = alpha(i)*alpha_bar(i-1);
end
% ... check approximations ...
sigma_tilde = [beta(1) sqrt( (1-alpha(2:end)).*(1-alpha_bar(1:end-1))./(1-alpha_bar(2:end)) )];
% ... exact mean and std ...
X_0 = [1;2];
% ... initial condition ...
mu_ex  = sqrt(alpha_bar(end))*X_0;
std_ex = sqrt(1 - alpha_bar(end)); 
Xh_0 = mvnrnd(mu_ex,std_ex.^2*eye(2),N)'; 
%% SDE setup
mu  = @(t) exp(-t^2/4)*X_0;
std = @(t) sqrt(1 - exp(-t^2/2));
s = @(x,t) -( x - mu(t) )./std(t).^2;
eps = @(x,t) -std(t)*s(x,t);
%% Euler-Maruyama method
for i = M:-1:1
   ti = i*dt;
%    Xh_0 = 1/sqrt(alpha(i))*( Xh_0 + (1-alpha(i))*s(Xh_0,ti) ) ...
%         + sqrt(beta(i))*randn(2,N);

   Xh_0 = 1/sqrt(alpha(i))*( Xh_0 - (1-alpha(i))/sqrt(1-alpha_bar(i))*eps(Xh_0,ti) ) ...
        + sigma_tilde(i)*randn(2,N);
   
%    Xh_0 = 1/sqrt(alpha(i))*Xh_0 - (1-alpha(i))/sqrt(1-alpha_bar(i))*eps(Xh_0,ti) + ...
%         + sigma_tilde(i)*randn(2,N);
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
disp(0*eye(2));
disp('numer.Cov  = '); disp(' ');
disp(cov_sde);