clc; clear all; close;
%% MNIST reverse DDPM process (SAMPLING)
%  X_k = sqrt( 1-beta_k )*X_{k-1} + sqrt( beta_k )*Z
%  X(0) = X0 (const.)
%  ------------------------------ 
%  mean = sqrt( bar{alpha}_k*X0 )
%  var  = 1 - bar{alpha}_k
%  ------------------------------ 
%% numerical setup
d  = 784;  % problem dimension
T  = 10;   % terminal time
M  = 1000; % number of iterations
dt = T/M;  % time step size
N  = 1;    % number of particles
% ... parameters in DDPM ...
beta      = (1:M)*dt^2;
alpha     = 1 - beta;
alpha_bar = zeros(1,M); alpha_bar(1) = alpha(1);
for i = 2:M
alpha_bar(i) = alpha(i)*alpha_bar(i-1);
end
sigma_tilde = [beta(1) sqrt( (1-alpha(2:end)).*(1-alpha_bar(1:end-1))./(1-alpha_bar(2:end)) )];
% ... exact mean (X0) ...
% X_0 = randn(d,1);
load X_0.mat;
% ... initial condition ...
% mu_ex  = sqrt(alpha_bar(end))*X_0;
% std_ex = sqrt(1 - alpha_bar(end)); 
mu_ex  = zeros(d,1);
std_ex = 1; 
Xh_0   = mvnrnd(mu_ex,std_ex.^2*eye(d),N)'; 
imshow(reshape(Xh_0,[28,28])); pause(1);
%% SDE setup
mu  = @(t) exp(-t^2/4)*X_0;
std = @(t) sqrt(1 - exp(-t^2/2));
s = @(x,t) -( x - mu(t) )./std(t).^2;
eps = @(x,t) -std(t)*s(x,t);
%% Euler-Maruyama method
for i = M:-1:1
   ti = i*dt;
%    Xh_0 = 1/sqrt(alpha(i))*( Xh_0 + (1-alpha(i))*s(Xh_0,ti) ) ...
%         + sqrt(beta(i))*randn(d,N);

   Xh_0 = 1/sqrt(alpha(i))*( Xh_0 - (1-alpha(i))/sqrt(1-alpha_bar(i))*eps(Xh_0,ti) ) ...
        + sigma_tilde(i)*randn(d,N);
   
%    Xh_0 = 1/sqrt(alpha(i))*Xh_0 - (1-alpha(i))/sqrt(1-alpha_bar(i))*eps(Xh_0,ti) + ...
%         + sigma_tilde(i)*randn(d,N);

   drawnow; imshow(reshape(Xh_0,[28,28]));
end
%% Compute mean and std from discrete data
mu_sde  = sum(Xh_0,2)/N;
cov_sde = cov(Xh_0')*(1-1/N);
%% Output
% disp('exact.mean = '); disp(' ');
% disp(X_0');
% disp('numer.mean = '); disp(' ');
% disp(mu_sde');
% disp('---------------------');
% disp('exact.Cov  = '); disp(' ');
% disp(0*eye(d));
% disp('numer.Cov  = '); disp(' ');
% disp(cov_sde);
norm(mu_sde-X_0,'inf')
% imshow(reshape(Xh_0,[28,28]));