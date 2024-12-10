clc; clear all; close all;
addpath('NN_library');
addpath('autodiff');
%% Learning the 1d score function via the O-U process
%  dXt  = -beta/2*dXt*dt + sqrt(beta)*dWt
%  X(0) = X0 ~ N( mu_0, sigma_0 )
%  --------------------------------------
%  score(X,t) = -( X - mu(t) )/( exp(-beta*t)*sigma_0^2 + sigma(t)^2 )
%  eps(X,t)   = -sigma(t)*score(X,t)
%  mu(t)      = exp(-beta/2*t)*mu_0
%  sigma(t)   = sqrt( 1 - exp(-beta*t) )
%% Global parameters
global n L xin yin act_fn sigmaT;
%% Network parameters
n      = [2 10 1];                    % dimension of each layer
m      = 5d3;                         % number of training data
m_test = 1d0*m;                       % number of test data
L      = length(n);                   % number of total layers
act_fn = "logsig";                    % activation function
totWb  = dot(n(1:L-1)+1,n(2:L));      % total number of weights and biases
%% LM parameters
maxit  = 1d3;                         % maximum number of epochs
TOL    = 1d-3;                        % tolerance
opt    = "chol";                      % optimizer
p0     = randn(totWb,1);              % initial guess
eta    = 1d0;                         % initial damping parameter
%% SDE parameters
T       = 10;
mu_0    = 3;
sigma_0 = 1;
beta    = 3;
mu      = @(X,t) exp(-beta/2*t).*X;
sigma   = @(t) sqrt( 1 - exp(-beta*t) );
score_exact = @(X,t) -( X - mu(mu_0,t) )./( exp(-beta*t)*sigma_0^2 + sigma(t).^2 );
eps_exact   = @(X,t) -sigma(t).*score_exact(X,t);
%% Problem setup
% ... Generate training data (Xt,t) ...
   % ... run SDE to generate Xt ...
%    X0    = mu_0 + sigma_0*randn(1,m);
   X0    = normrnd(mu_0,sigma_0,1,m);
   t     = sort( [T*rand(1,m-1) T] );
   noise = randn(1,m);
   Xt    = mu(X0,t) + sigma(t).*noise;
   sigmaT = sigma(t);
   % ... training points ...
   xin   = [Xt;t];
% ... target output ...
yin = noise;

% mu_sde  = sum(Xt)/m
% std_sde = sqrt( sum((Xt-mu_sde).^2)/m )

% mu(mu_0,T)
% sqrt( exp(-beta*T)*sigma_0^2 + sigma(T).^2 )
% scatter3(Xt,t,yin,20,yin,'filled');
% set(gca,'LineWidth',2,'FontSize',16);
% xlabel('Xt'); ylabel('t');
% return;
%% Display network parameters
disp(['Training points : ', num2str(length(xin))]);
disp(['# of Parameters : ', num2str(totWb)]);
%% Levenberg-Marquardt algorithm
loss    = zeros(maxit,1); % loss function
eta_rec = zeros(maxit,1); % damping parameter
epoch   = 1;              % initial step

while ( epoch <= maxit )
   % ... Computation of Jacobian matrix ...
   J = AutoDiffJacobianAutoDiff(@cost_vec, p0); J = -full(J);
   % ... Computation of the vector cost function ...
   res = cost_vec(p0);
   % ... Update p_{k+1} using LM algorithm ...
   if ( strcmp( opt, "svd" ) )
      % ... SVD ...
      [U,S,V] = svdecon(full(J));
      p = p0 + V*(U'.*(diag(S)./(diag(S).^2+eta)))*res(:);
   elseif ( strcmp( opt, "chol" ) )
      % ... Cholesky factorization ...
      R = chol(J'*J+eta*eye(totWb)); 
      p = p0 + R\(R'\(J'*res(:))); 
      p0 = p; res = cost_vec(p);
      p = p0 + R\(R'\(J'*res(:)));
   elseif ( strcmp( opt, "qr" ) )  
      % ... QR factorization ...
      dA = decomposition([J;sqrt(eta)*eye(totWb)]);
      p = p0 + dA\[res(:);zeros(totWb,1)];
      p0 = p; res = cost_vec(p);
      p = p0 + dA\[res(:);zeros(totWb,1)];
   end
   % ... Compute the cost function ...
   res = cost_vec(p0);
   loss(epoch) = sum( res(:).^2 );
   % ... monitor the decay of cost function ...
   eta_rec(epoch) = eta;
   % ... Damping parameter strategy ...
   if mod(epoch,5) == 0 && loss(epoch)<loss(epoch-1), eta = max(eta/1.5,1d-09); end
   if mod(epoch,5) == 0 && loss(epoch)>loss(epoch-1), eta = min(eta*2.0,1d+08); end
   % ... Break the loop if the certain condition is satisfied ...
   if loss(epoch) <= TOL, break; end
   % ... Next iteration loop ...
   epoch = epoch+1; p0 = p;
   % ... resampling ...
   if mod(epoch,10) == 0
      % ... Generate training data (Xt,t) ...
      % ... run SDE to generate Xt ... 
      X0    = mu_0 + sigma_0*randn(1,m);
      t     = sort( [T*rand(1,m-1) T] );
      noise = randn(1,m);
      Xt    = mu(X0,t) + sigma(t).*noise;
      sigmaT = sigma(t);
      % ... training points ... 
      xin   = [Xt;t];
      % ... target output ...
      yin = noise;
   end
end
loss( loss == 0 ) = [];
disp(['Training LOSS   : ', num2str(loss(end),'%.3e'),' (epoch = ', num2str(epoch),')']);

% ... ouput the training hostory ...
drawnow; loglog(eta_rec,'--','linewidth',1); hold on; loglog(loss,'linewidth',3);
loglog([1,length(loss)],[TOL TOL],'k:','linewidth',5); hold off;
title(['LOSS = ', num2str(loss(end),'%.3e'), ', \eta = ', num2str(eta,'%.3e')]);
set(gca,'fontsize',20,'linewidth',1); grid on; xlabel('training step'); axis([0 epoch TOL/1d1 max(loss)]);
%% Testing & Output
% ... Generate training data (Xt,t) ...
   % ... run SDE to generate Xt ... 
   X0    = mu_0 + sigma_0*randn(1,m_test);
   t     = sort( [T*rand(1,m_test-1) T] );
   noise = randn(1,m_test);
   Xt    = mu(X0,t) + sigma(t).*noise;
   % ... test points ... 
   x_test = [Xt;t];

%    x_test = xin;

   y_test = neural_network(x_test,p);
   s_test = score_exact(x_test(1,:),x_test(2,:));

figure(2);
subplot(1,2,1);
scatter(x_test(1,:),x_test(2,:),20,y_test,'filled'); colorbar; box on;
set(gca,'LineWidth',2,'FontSize',16);
title('$score_N(X_t,t)$','Interpreter','latex');
subplot(1,2,2);
scatter(x_test(1,:),x_test(2,:),20,s_test,'filled'); colorbar; box on;
set(gca,'LineWidth',2,'FontSize',16);
title('$score(X_t,t)$','Interpreter','latex');
%% Cost vector function
function f = cost_vec(p)
   global xin yin sigmaT;
   % ... Network function evaluation ...
   y = neural_network(xin,p);
   % ... output ...
   f = (sigmaT.*yin + y)'/sqrt(length(xin));
end
%% Jacobian of cost function
function J = Jacobian_cost(p)
   global n L xin act_fn totWb
   % ... preallocation ...
   J     = zeros(length(xin),totWb); % Jacobian matrix
   a     = cell(L,1);                % activation
   z     = cell(L,1);                % pre-activation
   delta = cell(L,1);                % delta
   dW    = cell(L,length(xin));      % derivative wrt weight
   db    = cell(L,length(xin));      % derivative wrt bias
   % ... convert parameter to weights and biases ...
   [W,b] = Param_2_Wb(p,n);
   % ... forward pass ...
   a{1} = xin;
   for l = 2:L-1
      z{l} = W{l}*a{l-1} + b{l};
      a{l} = activation( z{l}, act_fn );
   end
   a{L} = W{L}*a{L-1} + b{L};
   % ... Backward pass ...
%    delta{L} = dactivation( z{L}, act_fn(L) );
   delta{L} = ones( size(a{L}) );
   for l = L-1:-1:2
      delta{l} = dactivation( z{l}, act_fn ).*( W{l+1}'*delta{l+1} );   
   end
   % ... Jacobian marix ...
   for i = 1:length(xin)
      for l = 2:L
      dW{l,i} = delta{l}(:,i)*a{l-1}(:,i)';
      db{l,i} = delta{l}(:,i);
      end
      J(i,:) = Wb_2_Param({(dW{1:L,i})},{(db{1:L,i})},n);
   end
   
   J = J/sqrt(length(xin));
end
%% Neural network function evaluation
function y = neural_network(x,p)
   global n L act_fn
   % ... preallocation ...
   a = cell(L,1); % activation
   z = cell(L,1); % pre-activation
   % ... convert parameter to weights and biases
   [W,b] = Param_2_Wb(p,n);
   % ... forward pass ...
   a{1} = x;
   for l = 2:L-1
      z{l} = W{l}*a{l-1} + b{l};
      a{l} = activation( z{l}, act_fn );
   end
   a{L} = W{L}*a{L-1} + b{L};
   y = a{L};
end
%% Activation function
function y = activation(x,act_fn)
   if strcmp(act_fn,'logsig'),  y = logsig(x);  end
   if strcmp(act_fn,'tansig'),  y = tansig(x);  end
   if strcmp(act_fn,'poslin'),  y = poslin(x);  end
   if strcmp(act_fn,'purelin'), y = purelin(x); end
end
function y = dactivation(x,act_fn)
   if strcmp(act_fn,'logsig'),  y = logsig(x).*(1-logsig(x)); end
   if strcmp(act_fn,'tansig'),  y = 1-tansig(x).^2;           end
   if strcmp(act_fn,'poslin'),  y = ones(size(x));            end
   if strcmp(act_fn,'purelin'), y = ones(size(x));            end
end
function y = d2activation(x,act_fn)
   if strcmp(act_fn,'logsig'),  y = logsig(x).*(1-logsig(x)).*(1-2*logsig(x)); end
   if strcmp(act_fn,'tansig'),  y = (1-tansig(x).^2).*(-2*tansig(x));          end
   if strcmp(act_fn,'poslin'),  y = zeros(size(x));                            end
   if strcmp(act_fn,'purelin'), y = zeros(size(x));                            end
end