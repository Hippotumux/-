clc; clear all; close;
%% 2D reverse Ornstein-Uhenbeck process
%  dXt  = F*Xt*dt + G*dWt
%  X(0) = X0 ~ N(mu_0,SIGMA_0)
%  ------------------------------ 
%  mean(t) = exp(t*F)*mu_0
%  Cov(t)  = SIGMA(t) (solved by RK4)
%  score(x,t) = -inv(SIGMA(t))*( x - mean(t) )
%  ------------------------------ 
%% numerical setup
T  = 2;    % terminal time
M  = 1000; % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... construct F and G ...
F = randn(2,2);
F = -F*F';
G = randn(2,2);
G = G*G';
% ... exact mean and std at t = 0 ...
mu_0    = [1;2];
SIGMA_0 = randn(2,2);
SIGMA_0 = SIGMA_0'*SIGMA_0;
% ... initial condition ...
mu    = @(t) expm(t*F)*mu_0;
SIGMA = @(t) RK4_OU_covariance(t,F,G,SIGMA_0);
Xh_0  = mvnrnd(mu(T),SIGMA(T),N)'; 
%% Ornstein-Uhenbeck process
f = @(x,t) F*x;
g = @(x,t) G;
s = @(x,t) -SIGMA(t)\( x - mu(t) );
%% Euler-Maruyama method (backward)
for i = M:-1:1
   ti = i*dt;
   Xh_0 = Xh_0 + ( f(Xh_0,ti) - g(Xh_0,ti)*g(Xh_0,ti)'*s(Xh_0,ti) )*(-dt) ...
        + g(Xh_0,ti)*sqrt(dt)*randn(2,N);
end
%% Compute mean and cov from discrete data
mu_sde  = sum(Xh_0,2)/N;
cov_sde = cov(Xh_0')*(1-1/N);
%% Output
xx = mu_0(1)-5:0.1:mu_0(1)+5; yy = mu_0(2)-5:0.1:mu_0(2)+5;
[xx,yy] = meshgrid(xx,yy);
zz = mvnpdf([xx(:) yy(:)],mu_0',SIGMA_0');
zz = reshape(zz,[size(xx,1),size(yy,2)]);
imagesc(xx(:),yy(:),zz); hold on;
set(gca,'FontSize',16,'LineWidth',2,'YDir','normal');

plot(Xh_0(1,:),Xh_0(2,:),'w.','MarkerSize',4);
title('$\mathbf{X}_0 \sim \mathcal{N}(\mathbf{\mu}_0,\Sigma_0)$','interpreter','latex');

disp('exact.mean = '); disp(' ');
disp(mu_0');
disp('numer.mean = '); disp(' ');
disp(mu_sde');
disp('---------------------');
disp('exact.Cov  = '); disp(' ');
disp(SIGMA_0);
disp('numer.Cov  = '); disp(' ');
disp(cov_sde);