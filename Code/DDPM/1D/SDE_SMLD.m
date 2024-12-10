clc; clear all; close;
%% 1D forward Langevin Dynamics process
%  dXt  = sqrt( d( sigma(t)^2 )/dt )*dWt
%  X(0) = X0 (const.)
%  ------------------------------ 
%  sigma = sigma_min*( sigma_max/sigma_min )^(t/T)
%  mean = X0
%  var  = sigma_min^2*( ( sigma_max/sigma_min )^(2*t/T) - 1 )
%  ------------------------------ 
%% numerical setup
T  = 2;    % terminal time
M  = 100;  % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... parameters in f and g ...
sigma_min = 0.2;
sigma_max = 2;
% ... initial condition ...
X_0   = 2;
% ... exact mean and std at t = T ...
mu_ex  = X_0;
std_ex = sqrt( sigma_max^2 - sigma_min^2 );
%% SDE setup
f = @(x,t) 0;
g = @(x,t) sigma_min*( sigma_max/sigma_min )^(t/T)*sqrt( 2/T*log(sigma_max/sigma_min) );
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
set(gca,'FontSize',16,'LineWidth',2);
subplot(1,2,2);
plot(-0.1,Xh_0(:,M+1),'k.',normpdf(-10:0.1:10,mu_ex,std_ex),-10:0.1:10,'LineWidth',2);
axis off; 

disp(['exact.mean = ', num2str(mu_ex,'%.6f')]);
disp(['numer.mean = ', num2str(mu_sde,'%.6f')]);
disp('---------------------');
disp(['exact.std  = ', num2str(std_ex,'%.6f')]);
disp(['numer.std  = ', num2str(std_sde,'%.6f')]);