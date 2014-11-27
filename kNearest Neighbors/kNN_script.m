clear all

% Assume: 
% X_train contains the training points.
% Y_train contains the training labels.
% X_test contains the testing points.
% Y_test contains the testing labels.

Ks = [1, 5, 10, 50]; % Just the values of k of interest to be evaluated.

estimationType = 'classification'; % OR: 'regression'

% Computing errors per k for kNN based on leave-one-out applied to the training set.
errors = kNN_learning(X_train, Y_train, Ks, estimationType);

[~,i_min] = min(errors(:,2));
K = errors(i_min,1);

% Computing the estimation based on the optimal K.
Y_est = kNN_estimation(X_train, Y_train, X_test, K, estimationType);

error = mean(abs(Y_test-Y_est));