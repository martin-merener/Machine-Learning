% Script implementing Support Vector Machines Explained, by Tristan Fletcher
% Case solved: linear regression - Solution via QP and Kernels.

kernelFun = 'lap_kXY';
d = 1; % dimension of space
N = 20; % number of training points           

MaxIter_Data = 200; % Matlab default is 200

% Creating target function: targetFun
a1 = rand();
a2 = rand();
a3 = rand();

% Generating points
X_train = rand(N,d); % training points
Y_train = X_train(:,1).*(X_train(:,1)-a1).*(X_train(:,1)-a2).*(X_train(:,1)-a3).*(X_train(:,1)-1);

% Coefficients to solve the optimization problem:
C = 10;
epsilon = 0.1;
H = feval(kernelFun,X_train,X_train);
f = epsilon*ones(N,1)-Y_train;
Aeq = ones(N,1)';
beq = 0;
lb = repmat(-C,N,1);
ub = repmat(C,N,1);

% Optimization: THOR or DELL (Ryerson)
% options = optimoptions('quadprog');
% options = optimoptions(options,'Display', 'off');
% options = optimoptions(options,'Algorithm', 'interior-point-convex');
% options = optimoptions(options,'MaxIter', MaxIter_Data);
% [alpha,fval,exitflag,output,lambda] = quadprog(H,f,A,b,Aeq,beq,[],[],[],options);

% Optimization: TOSHIBA
options = optimset;
options = optimset(options,'Display', 'off');
options = optimset(options,'Algorithm', 'interior-point-convex');
%[alpha,fval,exitflag,output,lambda] = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);
[alpha,fval,exitflag,output,lambda] = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);

% Coefficients for the hypothesis:
%alpha_p = max(alpha,0);
%alpha_m = -min(alpha,0);

b1 = mean(Y_train - H*alpha);
Y_prime = H*alpha + b1;

S1 = find(abs(Y_prime-Y_train)<epsilon);
S2 = find(abs(alpha)>0.0001);
S3 = find(abs(alpha)<C);
S = intersect(S1,intersect(S2,S3));
n_S = length(S);

b2 = mean(Y_train(S) - feval(kernelFun,X_train(S,:),X_train)*alpha);
b3 = mean(Y_train(S)) - mean(feval(kernelFun,X_train(S,:),X_train)*alpha);

% Testing:
% Generating points
N_test = 1000;
X_test = rand(N_test,d); % training points
Y_test = X_test(:,1).*(X_test(:,1)-a1).*(X_test(:,1)-a2).*(X_test(:,1)-a3).*(X_test(:,1)-1);

Y_guess = feval(kernelFun,X_test,X_train)*alpha+b1;

hold off
scatter(X_test,Y_test,'black','.');
hold on
scatter(X_test,Y_guess,'red','.');
scatter(X_train,Y_train+0.01,'green','.');

mean(abs(Y_test-Y_guess))
