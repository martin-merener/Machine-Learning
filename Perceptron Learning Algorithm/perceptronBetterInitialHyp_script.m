% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%

% Testing PLA with uniformly distributed data, and a perfectly linear
% (random) target function.
% But instead of initializing with all zero weights, we initialize it with
% the optimal with respect to the linear model.

d = 2; % dimension of space
N = 10; % number of training points

nRuns = 1000; % number of runs, in each of which a new training set is simulated.
results = zeros(nRuns,3); 
results_alt = zeros(nRuns,3); 

for J = 1:nRuns

disp(J);
    
X = 2*rand(N,d)-1; % training points

% Creating target function: targetFun
Z = 2*rand(d,d)-1; % points that determine the hyperplane
P_in = Z(1,:);
G = Z(2:d,:)-repmat(P_in,d-1,1); % generators of the hyperplane's direction
P_out = 2*P_in; % point outside the hyperplane (?)
normal = P_out' - G'*pinv(G*G')*G*P_out'; % normal to hyperplane

Y = sign((X-repmat(P_in,N,1))*normal); % given a point x outside the plane, the sign of (x-P_in)*normal determines wheter x lies on the plane or not

% Now X, Y are given, so we define the PLA algorithm
maxIter = Inf;
[w, totIter] = perceptronPocket_learningAlgo(X,Y,maxIter); % returns weights that determines the hypothesis, and the total number of iterations
[w_alt, totIter_alt] = perceptronPocketBetterInitial_learningAlgo(X,Y,maxIter);

X_e = [ones(N,1), X];   % extended pts

% Error on train set. Initializing w as zero
Y_est = zeros(size(Y)); % estimated class
Y_est(X_e*w>=0) = 1;    
Y_est(X_e*w<0) = -1;
accuracy_train = mean(Y==Y_est); % accuracy

% Error on train set. Initializing w via LR
Y_est_alt = zeros(size(Y)); % estimated class
Y_est_alt(X_e*w_alt>=0) = 1;    
Y_est_alt(X_e*w_alt<0) = -1;
accuracy_train_alt = mean(Y==Y_est_alt); % accuracy

Ntest = 10000;
Xtest = 2*rand(Ntest,d)-1; % training points
Ytest = sign((Xtest-repmat(P_in,Ntest,1))*normal); % labels for testing points
Xtest_e = [ones(Ntest,1), Xtest];   % extended pts

% Error on train set. Initializing w as zero
Ytest_est = zeros(size(Ytest)); % estimated class
Ytest_est(Xtest_e*w>=0) = 1;
Ytest_est(Xtest_e*w<0) = -1;
accuracy_test = mean(Ytest==Ytest_est); % accuracy

% Error on train set. Initializing w via LR
Ytest_est_alt = zeros(size(Ytest)); % estimated class
Ytest_est_alt(Xtest_e*w_alt>=0) = 1;
Ytest_est_alt(Xtest_e*w_alt<0) = -1;
accuracy_test_alt = mean(Ytest==Ytest_est_alt); % accuracy

results(J,1) = totIter;
results(J,2) = accuracy_train;
results(J,3) = accuracy_test;

results_alt(J,1) = totIter_alt;
results_alt(J,2) = accuracy_train_alt;
results_alt(J,3) = accuracy_test_alt;

end

mean(results)
mean(results_alt)