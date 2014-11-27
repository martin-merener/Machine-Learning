% Multivariate Linear Regression

clear all

% Assume: 
% X_train contains the training points.
% Y_train contains the training labels.
% X_test contains the testing points.
% Y_test contains the testing labels.

% % DATA EXAMPLE
X_train = rand(10000,4);
X_test = rand(10000,4);
w_real = [1; 2; -3; -4];
w0 = 7;
Y_train = X_train*w_real+w0+normrnd(0,1,size(X_train,1),1);
Y_test = X_test*w_real+w0+normrnd(0,1,size(X_train,1),1);

% Computing the optimal coefficients for LR model.
X = X_train;
Y = Y_train;
X = [ones(size(X,1),1), X];
w_optimal = ((X'*X)\(X'))*Y; % Note that (X'*X)\(X') = inv(X'*X)*(X')
error_train = mean(sum((X*w_optimal-Y).^2,2));

% Computing the estimation based on the optimal w.
X = X_test;
X = [ones(size(X,1),1), X];
Y_est = X*w_optimal;
error_test = mean(abs(Y_test-Y_est));

disp([error_train, error_test]);