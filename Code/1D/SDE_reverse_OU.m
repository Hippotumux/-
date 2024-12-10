clc; clear all; close;
%% 1D reverse Ornstein-Uhenbeck process
%  dXt  = -beta*Xt*dt + sigma*dWt
%  X(0) = X0 ~ N(mu_0, sigma_0^2)
%  ------------------------------ 
%  mean  = mu_0*exp(-beta*t)
%  var   = sigma_0^2*exp(-2*beta*t) + sigma^2/(2*beta)*( 1 - exp(-2*beta*t) )
%  ------------------------------ 
%% numerical setup
T  = 2;    % terminal time
M  = 100;  % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... parameters in f and g ...
beta  = 1;
sigma = 0.5;
% ... exact mean and std at t = 0 ...
mu_0    = 2;
sigma_0 = 2;
% ... initial condition ...
mu  = @(t) mu_0*exp(-beta*t);
std = @(t) sqrt( exp(-2*beta*t)*sigma_0^2 + sigma^2/(2*beta)*(1-exp(-2*beta*t)) );
X_0 = normrnd(mu(T),std(T),N,1); 
%% Ornstein-Uhenbeck process
f = @(x,t) -beta*x;
g = @(x,t) sigma;
s = @(x,t) -( x - mu(t) )./std(t).^2;
%% Euler-Maruyama method (backward)
Xh_0 = zeros(N, M+1); Xh_0(:,M+1) = X_0;
for i = M:-1:1
   ti = i*dt;
   Xh_0(:,i) = Xh_0(:,i+1) + ( f(Xh_0(:,i+1),ti) - g(Xh_0(:,i+1),ti).^2.*s(Xh_0(:,i+1),ti) )*(-dt) ...
             + g(Xh_0(:,i+1),ti)*sqrt(dt)*randn(N,1);
end
%% Compute mean and std from discrete data
mu_sde  = sum(Xh_0(:,1))/N;
std_sde = sqrt( sum((Xh_0(:,1)-mu_sde).^2)/N );
%% Output
subplot(1,2,1);
plot(0:dt:T,Xh_0); ylim([-10,10]); xlabel('time'); ylabel('$\bar{X}_t$','Interpreter','latex');
set(gca,'FontSize',16,'LineWidth',2, 'XDir','reverse'); title('reverse');
subplot(1,2,2);
plot(-0.1,Xh_0(:,1),'k.',normpdf(-10:0.1:10,mu_0,sigma_0),-10:0.1:10,'LineWidth',2);
axis off;
% 
disp(['exact.mean = ', num2str(mu_0,'%.6f')]);
disp(['numer.mean = ', num2str(mu_sde,'%.6f')]);
disp('---------------------');
disp(['exact.std  = ', num2str(sigma_0,'%.6f')]);
disp(['numer.std  = ', num2str(std_sde,'%.6f')]);