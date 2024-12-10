clc; clear all; close;
%% 1D forward Ornstein-Uhenbeck process
%  dXt  = -beta*Xt*dt + sigma*dWt
%  X(0) = X0 ~ N(mu_0, sigma_0^2)
%  ------------------------------ 
%  mean = mu_0*exp(-beta*t)
%  var  = sigma_0^2*exp(-2*beta*t) + sigma^2/(2*beta)*( 1 - exp(-2*beta*t) )
%  ------------------------------ 
%% numerical setup
T  = 2;    % terminal time
M  = 100;  % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... parameters in f and g ...
beta  = 2;
sigma = 1;
% ... initial condition ...
mu_0    = 5;
sigma_0 = 0.3;
X_0     = normrnd(mu_0,sigma_0,N,1);
% ... exact mean and std at T ...
mu_ex  = mu_0*exp(-beta*T);
std_ex = sqrt( exp(-2*beta*T)*sigma_0^2 + sigma^2/(2*beta)*(1-exp(-2*beta*T)) );
%% SDE setup
f = @(x,t) -beta*x;
g = @(x,t) sigma;
%% Euler-Maruyama method
Xh_0 = zeros(N, M+1); Xh_0(:,1) = X_0;
for i = 1:M
%    Xh_0(:, i+1) = Xh_0(:, i) - beta*Xh_0(:, i)*dt + sigma*sqrt(dt)*randn(N,1);    
   ti = (i-1)*dt; 
   Xh_0(:,i+1) = Xh_0(:,i) + f(Xh_0(:,i),ti)*dt + g(Xh_0(:,i),ti)*sqrt(dt)*randn(N,1);
end
%% Compute mean and std from discrete data
mu_sde  = sum(Xh_0(:,M+1))/N;
std_sde = sqrt( sum((Xh_0(:,M+1)-mu_sde).^2)/N );
%% Output
subplot(1,2,1);
plot(0:dt:T,Xh_0); ylim([-10,10]); xlabel('time'); ylabel('X_t');
set(gca,'FontSize',12,'LineWidth',2);
subplot(1,2,2);
plot(-0.1,Xh_0(:,M+1),'k.',normpdf(-10:0.1:10,mu_ex,std_ex),-10:0.1:10,'LineWidth',2);
axis off;

disp(['exact.mean = ', num2str(mu_ex,'%.6f')]);
disp(['numer.mean = ', num2str(mu_sde,'%.6f')]);
disp('---------------------');
disp(['exact.std  = ', num2str(std_ex,'%.6f')]);
disp(['numer.std  = ', num2str(std_sde,'%.6f')]);