clc; clear all; close;
%% 1D forward DDPM process
%  X_k = sqrt( 1-beta_k )*X_{k-1} + sqrt( beta_k )*Z
%  X(0) = X0 (const.)
%  ------------------------------ 
%  mean = sqrt( bar{alpha}_k*X0 )
%  var  = 1 - bar{alpha}_k
%  ------------------------------ 
%% numerical setup
T  = 2;    % terminal time
M  = 100;  % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... parameters in f and g ...
beta      = (1:M)*dt^2;
alpha     = 1 - beta;
alpha_bar = zeros(1,M); alpha_bar(1) = alpha(1);
for i = 2:M
alpha_bar(i) = alpha(i)*alpha_bar(i-1);
end
% std = sqrt( 1 - exp( -0.5*( (1:M)*dt ).^2 ) );
% b = exp( -0.5*( (1:M)*dt ).^2 );

% sigma = sqrt( (1-alpha(2:end)).*(1-alpha_bar(1:end-1))./(1-alpha_bar(2:end)) );
% ... initial condition ...
X_0   = 5;
% ... exact mean and std ...
mu_ex  = sqrt(alpha_bar(end))*X_0;
std_ex = sqrt(1 - alpha_bar(end));
%% SDE setup
% f = @(x,t) -beta(t)/2*x;
% g = @(x,t) sigma(t);
%% Euler-Maruyama method
Xh_0 = zeros(N, M+1); Xh_0(:,1) = X_0;
for i = 1:M    
   ti = (i-1)*dt; 
   Xh_0(:,i+1) = sqrt( 1 - beta(i) )*Xh_0(:,i) + sqrt( beta(i) )*randn(N,1);
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