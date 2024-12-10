clc; clear all; close;
%% 1D forward stochastic process
%  dXt  = mu*dt + sigma*dWt
%  X(0) = X0 (const.)
%  ------------------------- 
%  mean = X0 + mu*t
%  var  = sigma^2*t
%  ------------------------- 
%% numerical setup
T  = 2;    % terminal time
M  = 100;  % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... parameters in f and g ...
mu    = 3;
sigma = 0.5;

% ... initial condition ...
X_0   = 0;

% ... exact mean and std at T ...
mu_ex  = X_0 + mu*T;
std_ex = sigma*sqrt(T);
%% SDE setup
f = @(x,t) mu;
g = @(x,t) sigma;
%% Euler-Maruyama method
Xh_0 = zeros(N, M+1); Xh_0(:,1) = X_0;
for i = 1:M
   %    Xh_0(:,i+1) = Xh_0(:,i) + mu*dt + sigma*sqrt(dt)*randn(N,1); 
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
ylim([-10,10]); axis off; 
%
disp(['exact.mean = ', num2str(mu_ex,'%.6f')]);
disp(['numer.mean = ', num2str(mu_sde,'%.6f')]);
disp('---------------------');
disp(['exact.std  = ', num2str(std_ex,'%.6f')]);
disp(['numer.std  = ', num2str(std_sde,'%.6f')]);