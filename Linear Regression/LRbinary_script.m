% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%

% In this script we simulate data sets and use LR to classify (binary)

d = 2; % space dimension
N = 100; % number of training points
X = 2*rand(N,d)-1; % training points

% Creating target function (random)
Z = 2*rand(d,d)-1; % points that determine the hyperplane: d points determine and hyperplane in dimension d.
P_in = Z(1,:); % a point in the hyperplane
G = Z(2:d,:)-repmat(P_in,d-1,1); % generators of the hyperplane's direction
P_out = 2*P_in; % point outside the hyperplane (?)
normal = P_out' - G'*pinv(G*G')*G*P_out'; % normal to hyperplane

pre_Y = (X-repmat(P_in,N,1))*normal; % given a point x outside the plane, the sign of (x-P_in)*normal determines wheter x lies on the plane or not
Y = zeros(size(pre_Y));
Y(pre_Y>=0) = 1;
Y(pre_Y<0) = -1;

% Now X, Y are given, so compute LR
X_e = [ones(size(X,1),1), X];
w = ((X_e'*X_e)\(X_e'))*Y;

Y_est = zeros(size(Y)); % estimated class
Y_est(X_e*w>=0) = 1;    
Y_est(X_e*w<0) = -1;
accuracy_train = mean(Y==Y_est); % accuracy

Ntest = 1000;
Xtest = 2*rand(Ntest,d)-1; % training points
pre_Ytest = (Xtest-repmat(P_in,Ntest,1))*normal; % labels for testing points
Ytest = zeros(size(pre_Ytest));
Ytest(pre_Ytest>=0) = 1;
Ytest(pre_Ytest<0) = -1;

Xtest_e = [ones(Ntest,1), Xtest];   % extended pts
Ytest_est = zeros(size(Ytest)); % estimated class
Ytest_est(Xtest_e*w>=0) = 1;
Ytest_est(Xtest_e*w<0) = -1;
accuracy_test = mean(Ytest==Ytest_est); % accuracy
