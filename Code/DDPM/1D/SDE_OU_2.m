clc; clear all; close;
%% 1D forward Ornstein-Uhenbeck process
%  dXt  = -beta(t)/2*Xt*dt + sqrt(beta(t))*dWt
%  X(0) ~ N(mu_0,sigma_0^2)
%  ------------------------------ 
%  beta = t
%  mean = exp(-T^2/4)*mu_0
%  var  = exp(-T^2/2)*sigma_0^2 + 1 - exp(-T^2/2)
%  ------------------------------ 
%% numerical setup
T  = 1;    % terminal time
M  = 100;  % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... parameters in f and g ...
beta  = @(t) t;
sigma = @(t) sqrt(beta(t));
% ... initial condition ...
mu_0    = 5;
sigma_0 = 0.3;
X_0     = normrnd(mu_0,sigma_0,N,1);
% ... exact mean and std at t = T ...
mu_ex  = exp(-T^2/4)*mu_0;
std_ex = sqrt( exp(-T^2/2)*sigma_0^2 + 1 - exp(-T^2/2) );
%% SDE setup
f = @(x,t) -beta(t)/2*x;
g = @(x,t) sigma(t);
%% Euler-Maruyama method
Xh_0 = zeros(N, M+1); Xh_0(:,1) = X_0;
for i = 1:M
   ti = (i-1)*dt; 
   Xh_0(:,i+1) = Xh_0(:,i) + f(Xh_0(:,i),ti)*dt + g(Xh_0(:,i),ti)*sqrt(dt)*randn(N,1);
end
%% Compute mean and std from discrete data
mu_sde  = sum(Xh_0(:,M+1))/N;
std_sde = sqrt( sum((Xh_0(:,M+1)-mu_sde).^2)/N );
%% Output
subplot(1,2,1);
plot(0:dt:T,Xh_0); ylim([-10,10]); xlabel('time'); ylabel('X_t');
set(gca,'FontSize',16,'LineWidth',2);
subplot(1,2,2);
plot(-0.1,Xh_0(:,M+1),'k.',normpdf(-10:0.1:10,mu_ex,std_ex),-10:0.1:10,'LineWidth',2);
axis off; 

disp(['exact.mean = ', num2str(mu_ex,'%.6f')]);
disp(['numer.mean = ', num2str(mu_sde,'%.6f')]);
disp('---------------------');
disp(['exact.std  = ', num2str(std_ex,'%.6f')]);
disp(['numer.std  = ', num2str(std_sde,'%.6f')]);