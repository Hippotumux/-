function S = RK4_OU_covariance(endT,F,G,S0)
%% NUMERICAL PARAMETER
dt = 1d-2;
nt = ceil(endT/dt);
H  = @(t,S) F*S + S*F' + G*G';
%% INITIAL CONDITION
S = S0;
%% MAIN LOOP
for it = 1:nt
   k1 = dt*H((it-1)*dt    ,S);
   k2 = dt*H((it-1+1/2)*dt,S+k1/2);
   k3 = dt*H((it-1+1/2)*dt,S+k2/2);
   k4 = dt*H((it-1+1)*dt  ,S+k3);
   S  = S + (k1+2*k2+2*k3+k4)/6;
end
end