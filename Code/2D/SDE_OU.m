clc; clear all; close;
%% 2D forward stochastic process
%  dXt  = F*Xt*dt + G*dWt
%  X(0) = X0 ~ N(mu_0,SIGMA_0)
%  ----------------------------- 
%  mean = exp(t*F)*mu_0
%  Cov  = solved by RK4
%  -----------------------------
%% numerical setup
T  = 2;    % terminal time
M  = 100;  % number of iterations
dt = T/M;  % time step size
N  = 2000; % number of particles
% ... construct F and G ...
F = randn(2,2);
F = -F*F';
G = randn(2,2);
G = G*G';
% ... initial condition ...
mu_0    = [1;2];
SIGMA_0 = randn(2,2);
SIGMA_0 = SIGMA_0'*SIGMA_0;
Xh_0 = mvnrnd(mu_0,SIGMA_0,N)';
% ... exact mean and cov at t = T ...
mu_ex  = expm(T*F)*mu_0;
cov_ex = RK4_OU_covariance(T,F,G,SIGMA_0);
%% SDE setup
f = @(x,t) F*x;
g = @(x,t) G;
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
xx = mu_ex(1)-5:0.1:mu_ex(1)+5; yy = mu_ex(2)-5:0.1:mu_ex(2)+5;
[xx,yy] = meshgrid(xx,yy);

zz = mvnpdf([xx(:) yy(:)],mu_ex',cov_ex');
zz = reshape(zz,[size(xx,1),size(yy,2)]);
% surf(xx,yy,zz,'FaceAlpha',0.8,'EdgeColor','none');

imagesc(xx(:),yy(:),zz); hold on;
set(gca,'FontSize',16,'LineWidth',2,'YDir','normal');
plot(Xh_0(1,:),Xh_0(2,:),'w.','MarkerSize',4);
title('$\mathbf{X}_t \sim \mathcal{N}(\mathbf{\mu}_t,\Sigma_t)$','interpreter','latex');
%
disp('exact.mean = '); disp(' ');
disp(mu_ex');
disp('numer.mean = '); disp(' ');
disp(mu_sde');
disp('---------------------');
disp('exact.Cov  = '); disp(' ');
disp(cov_ex);
disp('numer.Cov  = '); disp(' ');
disp(cov_sde);