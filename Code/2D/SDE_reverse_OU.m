clc; clear all; close;
%% 2D reverse Ornstein-Uhenbeck process
%  dXt  = -beta*Xt*dt + sigma*dWt
%  X(0) = X0 ~ N(mu_0, sigma_0^2)
%  ------------------------------ 
%  mean(t) = mu_0*exp(-beta*t)
%  Cov(t)  = sigma_0^2*exp(-2*beta*t) + sigma^2/(2*beta)*( 1 - exp(-2*beta*t) )
%  score(x,t) = -(x-mean(t))/Cov(t)
%  ------------------------------ 
%% numerical setup
T  = 2;    % terminal time
M  = 100;  % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... parameters in f and g ...
beta  = 1;
sigma = 5;
% ... exact mean and std at t = 0 ...
mu_0    = [1;2];
sigma_0 = 2;
% ... initial condition ...
mu  = @(t) mu_0*exp(-beta*t);
std = @(t) sqrt( exp(-2*beta*t)*sigma_0^2 + sigma^2/(2*beta)*(1-exp(-2*beta*t)) );
Xh_0 = mvnrnd(mu(T),std(T).^2*eye(2),N)'; 
%% Ornstein-Uhenbeck process
f = @(x,t) -beta*x;
g = @(x,t) sigma;
s = @(x,t) -( x - mu(t) )./std(t).^2;
%% Euler-Maruyama method (backward)
for i = M:-1:1
   ti = i*dt;
   Xh_0 = Xh_0 + ( f(Xh_0,ti) - g(Xh_0,ti).^2.*s(Xh_0,ti) )*(-dt) ...
        + g(Xh_0,ti)*sqrt(dt)*randn(2,N);
end
%% Compute mean and cov from discrete data
mu_sde  = sum(Xh_0,2)/N;
cov_sde = cov(Xh_0')*(1-1/N);
%% Output
xx = mu_0(1)-5:0.1:mu_0(1)+5; yy = mu_0(2)-5:0.1:mu_0(2)+5;
[xx,yy] = meshgrid(xx,yy);
zz = mvnpdf([xx(:) yy(:)],mu_0',sigma_0^2*eye(2)');
zz = reshape(zz,[size(xx,1),size(yy,2)]);
imagesc(xx(:),yy(:),zz); hold on;
set(gca,'FontSize',16,'LineWidth',2,'YDir','normal');

plot(Xh_0(1,:),Xh_0(2,:),'w.','MarkerSize',4);
title('$\mathbf{X}_0 \sim \mathcal{N}(\mathbf{\mu}_0,\sigma_0^2 I)$','interpreter','latex');
%
disp('exact.mean = '); disp(' ');
disp(mu_0');
disp('numer.mean = '); disp(' ');
disp(mu_sde');
disp('---------------------');
disp('exact.Cov  = '); disp(' ');
disp(sigma_0^2*eye(2));
disp('numer.Cov  = '); disp(' ');
disp(cov_sde);