numICs = 5000;
filenamePrefix = 'Duffing';

delta = 0.5;
beta = -1;
alpha = 1;
f = @(t,x)( 0.5 * [2*x(2,:) ; -delta*2*x(2,:) - 2*x(1,:)*(beta+alpha*(2*x(1,:)).^2)] ); % Uncontrolled
f_u = @(t,x,u)( 0.5 * [2*x(2,:) ; -delta*2*x(2,:) - 2*x(1,:).*(beta+alpha*(2*x(1,:)).^2) + u]  ); % Controlled

n = 2; % Number of states
m = 1; % Number of control inputs
% Sampling period
deltaT = 0.01;
trajLen = 800; % Lenght of tracetories
Ntraj = 100; % Number of trajectories



X_test = DuffingFn(trajLen,Ntraj, n,f_u,deltaT);
filename_test = strcat(filenamePrefix, '_test_x.csv');
dlmwrite(filename_test, X_test, 'precision', '%.14f')



X_val = DuffingFn(trajLen,2*Ntraj, n,f_u,deltaT);
filename_val = strcat(filenamePrefix, '_val_x.csv');
dlmwrite(filename_val, X_val, 'precision', '%.14f')

for j = 1:6
	seed = 2+j;
	X_train = DuffingFn(trajLen,7*Ntraj, n,f_u,deltaT);
	filename_train = strcat(filenamePrefix, sprintf('_train%d_x.csv', j));
	dlmwrite(filename_train, X_train, 'precision', '%.14f')
end