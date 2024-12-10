clc; clear all; close;
%% 1D reverse Ornstein-Uhenbeck process
%  dXt  = 
%  X(0) ~ N( mu_0, sigma_0^2 )
%  ------------------------------ 
%  mean  = 
%  var   = 
%  ------------------------------ 
%% numerical setup
T  = 10;   % terminal time
M  = 1000; % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... exact mean and std at t = 0 ...
mu_0    = 3;
sigma_0 = 1;
% ... initial condition ...
mu  = @(t) exp(-t^2/4)*mu_0;
std = @(t) sqrt( exp(-t^2/2)*sigma_0^2 + 1 - exp(-t^2/2) );
Xh_0 = zeros(N, M+1); Xh_0(:,M+1) = normrnd(mu(T),std(T),N,1);
%% reverse SDE process
beta = @(t) t;
f     = @(x,t) -beta(t)/2*x;
g     = @(x,t) sqrt(beta(t));
s     = @(x,t) -( x - mu(t) )./std(t).^2;
sigma = @(t) sqrt( 1 - exp(-t^2/2) );
eps   = @(x,t) -sigma(t)*s(x,t);
%% Euler-Maruyama method (backward)
for i = M:-1:1
   ti = i*dt;
%    Xh_0(:,i) = Xh_0(:,i+1) + ( f(Xh_0(:,i+1),ti) - g(Xh_0(:,i+1),ti).^2.*s(Xh_0(:,i+1),ti) )*(-dt) ...
%              + g(Xh_0(:,i+1),ti)*sqrt(dt)*randn(N,1);

   Xh_0(:,i) = ( 1 + 1/2*beta(ti)*dt )*Xh_0(:,i+1) - beta(ti)*dt/sigma(ti)*eps(Xh_0(:,i+1),ti) + ...
               + sqrt( beta(ti)*dt )*randn(N,1);

%    Xh_0(:,i) = 1/sqrt( 1 - beta(ti)*dt )*Xh_0(:,i+1) - beta(ti)*dt/std(ti)*eps(Xh_0(:,i+1),ti) + ...
%                + sqrt( beta(ti)*dt )*randn(N,1);

end
%% Compute mean and std from discrete data
mu_sde  = sum(Xh_0(:,1))/N;
std_sde = sqrt( sum((Xh_0(:,1)-mu_sde).^2)/N );
%% Output
subplot(1,2,1);
plot(0:dt:T,Xh_0); ylim([-10,10]); xlabel('time'); ylabel('$\bar{X}_t$','Interpreter','latex');
set(gca,'FontSize',16,'LineWidth',2, 'XDir','reverse'); title('reverse');
subplot(1,2,2);
plot(-0.1,Xh_0(:,1),'k.',normpdf(-10:0.1:10,mu_0,std(0)),-10:0.1:10,'LineWidth',2);
axis off;
% 
disp(['exact.mean = ', num2str(mu_0,'%.6f')]);
disp(['numer.mean = ', num2str(mu_sde,'%.6f')]);
disp('---------------------');
disp(['exact.std  = ', num2str(std(0),'%.6f')]);
disp(['numer.std  = ', num2str(std_sde,'%.6f')]);