clc; clear all; close;
%% 2D forward stochastic process
%  dXt  = SIGMA^(1/2)*dWt
%  X(0) = X0 (const.)
%  ------------------------- 
%  mean = X0
%  Cov  = t*SIGMA
%  ------------------------- 
%% numerical setup
T     = 2;    % terminal time
M     = 100;  % number of iterations
dt    = T/M;  % time step size
N     = 2000; % number of particles
% ... construct SIGMA and SIGMA^(1/2) ...
A       = randn(2,2);
[~,S,V] = svd(A);
SIGMA   = V*S.^2*V';
SIGMA_half = V*S*V';
% ... initial condition ...
X_0 = [1;2];
Xh_0 = zeros(2,N) + X_0;
% ... exact mean and cov at t = T ...
mu_ex  = X_0;
cov_ex = SIGMA*T;
%% SDE setup
f = @(x,t) 0;
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
set(gca,'FontSize',16,'LineWidth',2,'YDir','normal')
plot(Xh_0(1,:),Xh_0(2,:),'w.','MarkerSize',4);
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