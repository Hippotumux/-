clc; clear all; close;
%% 1D reverse DDPM process
%  X_k = sqrt( 1-beta_k )*X_{k-1} + sqrt( beta_k )*Z
%  X(0) = X0 (const.)
%  ------------------------------ 
%  mean = sqrt( bar{alpha}_k*X0 )
%  var  = 1 - bar{alpha}_k
%  ------------------------------ 
%% numerical setup
T  = 10;    % terminal time
M  = 1000;  % number of iterations
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
% std = sqrt( 1 - exp( -0.5*( (1:M)*dt ).^2 ) );
sigma_tilde = [beta(1) sqrt( (1-alpha(2:end)).*(1-alpha_bar(1:end-1))./(1-alpha_bar(2:end)) )];
% ... exact mean and std ...
X_0 = 2;
% ... initial condition ...
mu_ex  = sqrt(alpha_bar(end))*X_0;
std_ex = sqrt(1 - alpha_bar(end)); 
%% SDE setup
mu  = @(t) exp(-t^2/4)*X_0;
std = @(t) sqrt(1 - exp(-t^2/2));
s = @(x,t) -( x - mu(t) )./std(t).^2;
eps = @(x,t) -std(t)*s(x,t);
%% Euler-Maruyama method
Xh_0 = zeros(N, M+1); Xh_0(:,M+1) = normrnd(mu_ex,std_ex,N,1);

for i = M:-1:1
   ti = i*dt;
%    Xh_0(:,i) = 1/sqrt(alpha(i))*( Xh_0(:,i+1) + (1-alpha(i))*s(Xh_0(:,i+1),ti) ) ...
%              + sqrt(beta(i))*randn(N,1);

   Xh_0(:,i) = 1/sqrt(alpha(i))*( Xh_0(:,i+1) - (1-alpha(i))/sqrt(1-alpha_bar(i))*eps(Xh_0(:,i+1),ti) ) ...
             + sigma_tilde(i)*randn(N,1);
   
%    Xh_0(:,i) = 1/sqrt(alpha(i))*Xh_0(:,i+1) - (1-alpha(i))/sqrt(1-alpha_bar(i))*eps(Xh_0(:,i+1),ti) + ...
%                + sigma_tilde(i)*randn(N,1);
end
%% Compute mean and std from discrete data
mu_sde  = sum(Xh_0(:,1))/N;
std_sde = sqrt( sum((Xh_0(:,1)-mu_sde).^2)/N );
%% Output
% subplot(1,2,1);
plot(0:dt:T,Xh_0); ylim([-10,10]); xlabel('time'); ylabel('$\bar{X}_t$','Interpreter','latex');
set(gca,'FontSize',16,'LineWidth',2, 'XDir','reverse'); title('reverse');
% subplot(1,2,2);
% plot(-0.1,Xh_0(:,M+1),'k.',normpdf(-10:0.1:10,mu_ex,std_ex),-10:0.1:10,'LineWidth',2);
% axis off; 

disp(['exact.mean = ', num2str(X_0,'%.6f')]);
disp(['numer.mean = ', num2str(mu_sde,'%.6f')]);
disp('---------------------');
disp(['exact.std  = ', num2str(0,'%.6f')]);
disp(['numer.std  = ', num2str(std_sde,'%.6f')]);