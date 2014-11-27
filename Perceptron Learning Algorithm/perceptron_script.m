% Testing PLA with uniformly distributed data, and a perfectly linear
% (random) target function.

d = 3; % dimension of space

N = 1000; % number of training points

nRuns = 50; % number of runs, in each of which a new training set is simulated.
results = zeros(nRuns,2); 

for J = 1:nRuns

X = 2*rand(N,d)-1; % training points

% Creating target function: targetFun
Z = 2*rand(d,d)-1; % points that determine the hyperplane
P_in = Z(1,:);
G = Z(2:d,:)-repmat(P_in,d-1,1); % generators of the hyperplane's direction
P_out = 2*P_in; % point outside the hyperplane (?)
normal = P_out' - G'*pinv(G*G')*G*P_out'; % normal to hyperplane

Y = sign((X-repmat(P_in,N,1))*normal); % given a point x outside the plane, the sign of (x-P_in)*normal determines wheter x lies on the plane or not

% ptsInPlane = (2*rand(1000,d-1)-1)*G+repmat(P_in,1000,1);
% X_pos = X(Y>0,:);
% X_neg = X(Y<0,:);     
% hold off
% scatter(ptsInPlane(:,1),ptsInPlane(:,2),'black');
% % scatter3(ptsInPlane(:,1),ptsInPlane(:,2),ptsInPlane(:,3),'black');
% hold on
% scatter(X_pos(:,1),X_pos(:,2),'blue');
% scatter(X_neg(:,1),X_neg(:,2),'red');
% % scatter3(X_pos(:,1),X_pos(:,2),X_pos(:,3),'blue');
% % scatter3(X_neg(:,1),X_neg(:,2),X_neg(:,3),'red');

% Now X, Y are given, so we define the PLA algorithm
maxIter = Inf;
[w, totIter] = perceptronPocket_learningAlgo(X,Y,maxIter); % returns weights that determines the hypothesis, and the total number of iterations

X_e = [ones(N,1), X];   % extended pts
Y_est = zeros(size(Y)); % estimated class
Y_est(X_e*w>=0) = 1;    
Y_est(X_e*w<0) = -1;
accuracy_train = mean(Y==Y_est); % accuracy

Ntest = 10000;
Xtest = 2*rand(Ntest,d)-1; % training points
Ytest = sign((Xtest-repmat(P_in,Ntest,1))*normal); % labels for testing points

Xtest_e = [ones(Ntest,1), Xtest];   % extended pts
Ytest_est = zeros(size(Ytest)); % estimated class
Ytest_est(Xtest_e*w>=0) = 1;
Ytest_est(Xtest_e*w<0) = -1;
accuracy_test = mean(Ytest==Ytest_est); % accuracy

results(J,1) = totIter;
results(J,2) = 1-accuracy_test;

end

averageIteration = mean(results(:,1))
averageProbError = mean(results(:,2))
