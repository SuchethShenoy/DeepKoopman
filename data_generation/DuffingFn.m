
function X = DuffingFn(trajLen,Ntraj, n,f_u,deltaT)

%Runge-Kutta 4
k1 = @(t,x,u) (  f_u(t,x,u) );
k2 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT/2,u) );
k3 = @(t,x,u) ( f_u(t,x + k2(t,x,u)*deltaT/2,u) );
k4 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT,u) );
f_ud = @(t,x,u) ( x + (deltaT/6) * ( k1(t,x,u) + 2*k2(t,x,u) + 2*k3(t,x,u) + k4(t,x,u)  )   );

%% Collect trajectories
Traj = cell(1,Ntraj); % Cell array of trajectories
for j = 1:Ntraj
    %textwaitbar(j, Ntraj, "Collecting data without control")    
    xx = randn(n,1);
    xx = xx / norm(xx); % Unitial conditions on unit circle
    for i = 1:trajLen-1
        xx = [xx f_ud(0,xx(:,end),0)];
    end
    Traj{j} = xx;
end

% Vectorize data
F_vec = cell(1,n);
for i = 1:n
    F_vec{i} = [];
    for j = 1:numel(Traj)
        F_vec{i} = [F_vec{i} ; Traj{j}(i,:).'];
    end
end
X = F_vec;
