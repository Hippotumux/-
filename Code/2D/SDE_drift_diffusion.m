clc; clear all; close;
%% 2D forward stochastic process
%  dXt  = f*dt + SIGMA^(1/2)*dWt
%  X(0) = X0 ~ N(mu_0,SIGMA_0)
%  ----------------------------- 
%  mean = mu_0 + t*f
%  Cov  = SIGMA_0 + t*SIGMA
%  -----------------------------
%% numerical setup
T     = 2;    % terminal time
M     = 100;  % number of iterations
dt    = T/M;  % time step size
N     = 2000; % number of particles
% ... construct f, SIGMA and SIGMA^(1/2) ...
f_vec   = randn(2,1);
A       = randn(2,2);
[~,S,V] = svd(A);
SIGMA   = V*S.^2*V';
SIGMA_half = V*S*V';
% ... initial condition ...
mu_0    = [1;2];
SIGMA_0 = randn(2,2);
SIGMA_0 = SIGMA_0'*SIGMA_0;
Xh_0 = mvnrnd(mu_0,SIGMA_0,N)';
% ... exact mean and cov at t = T ...
mu_ex  = mu_0 + T*f_vec;
cov_ex = SIGMA_0 + SIGMA*T;
%% SDE setup
f = @(x,t) f_vec;
g = @(x,t) SIGMA_half;
%% Euler-Maruyama method
for i = 1:M
%    Xh_0 = Xh_0 + SIGMA_half*sqrt(dt)*randn(2,N);
   ti = (i-1)*dt;
   Xh_0 = Xh_0 + f(Xh_0,ti)*dt + g(Xh_0,ti)*sqrt(dt)*randn(2,N);
end
%% Compute mean and cov from discrete data
mu_sde  = sum(Xh_0,2)/N;
cov_sde = cov(Xh_0')*(1-1/N);
%% Output
% plot3(Xh_0(1,:),Xh_0(2,:),-0.1*ones(1,N),'r.'); hold on;
%
xx = mu_sde(1)-5:0.1:mu_sde(1)+5; yy = mu_sde(2)-5:0.1:mu_sde(2)+5;
[xx,yy] = meshgrid(xx,yy);

zz = mvnpdf([xx(:) yy(:)],mu_ex',cov_ex');
zz = reshape(zz,[size(xx,1),size(yy,2)]);
% surf(xx,yy,zz,'FaceAlpha',0.8,'EdgeColor','none');

imagesc(xx(:),yy(:),zz); hold on;
set(gca,'FontSize',16,'LineWidth',2,'YDir','normal');
plot(Xh_0(1,:),Xh_0(2,:),'w.','MarkerSize',4);
title('$\mathbf{X}_t \sim \mathcal{N}(\mathbf{\mu}_0,\Sigma_0 + t\Sigma)$','interpreter','latex');
%
disp('exact.mean = ');
disp(mu_ex');
disp('numer.mean = ');
disp(mu_sde');
disp('---------------------');
disp('exact.Cov  = ');
disp(cov_ex);
disp('numer.Cov  = ');
disp(cov_sde);