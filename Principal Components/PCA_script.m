% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%

% PCA is a dimensionality reduction algorithm.
% It finds a lower dimensional surface (plane) on which the data
% points are projected. The found surface is so that it minimizes the square
% of the distances (orthogonal projections) from the training points.
% The dimension of the surface is chosen to be minimal so that the
% sum of the errors from the point to their approximations is small enough
% relatively: sum(||x^(i) - x^(i)_approx||^2)/sum(||x^(i)||^2) < 0.01
% The algorithm finds the direction vectors determining the surface.
% The reduced points are in a lower dimensional space (surface).

clear all

% X_train contains the points to determine how to reduce the data. Each point is a row.
% X_to_reduce contains the data to be reduced (testing points, valid points, etc).

X_train = rand(100,10); % for example
m = size(X_train,1);

% Preprocessing: subtract the mean of each feature and divide by std
X_train_mean0 = X_train - ones(m,1)*mean(X_train); % The columns have mean 0.
X_norm = X_train_mean0*diag(1./std(X_train_mean0));% The columns have mean 0 and standard deviation 1. 
clear X_train_mean0 X_train

% PCA
cov_xx = (X_norm'*X_norm/m); 
[U,S,V] = svd(cov_xx); % U contains the principal components

% We look at the variance retained
S_diag = diag(S);
SumDiag = sum(S_diag);
k = 1;
var_ret = 0.99; % could be also 0.95.
while sum(S_diag(1:k))/SumDiag < var_ret  % this will give the smallest k such that sum(S_diag(1:k))/sum(S_diag) >= var_ret 
    k = k+1; % so this loop is getting the first k for which the variance retained is at least var_ret.
end
% Since: 1-sum(S_diag(1:k))/sum(S_diag) = 1-varianceRetainedByFirst_k_components = sum(||x^(i) - x^(i)_approx||^2)/sum(||x^(i)||^2)
% what this is doing is finding the minimum k (dimension of surface to which points are projected) so that:
% sum(||x^(i) - x^(i)_approx||^2)/sum(||x^(i)||^2) < 1-var_red
principal_k_components = U(:,1:k); % the first k columns of U are the principal components
X_to_reduce = rand(100,10); % If these were to be reduced...
X_reduced = X_to_reduce*principal_k_components; % these are the points after dimensionality reduction
X_approx = X_reduced*principal_k_components'; % this are the reduced points reconstructed

% HOW TO USE PCA TO SPEED UP A SUPERVISED LEARNING ALGORITHM
% We start with: training set [X_train,y], validation set [X_val,y_val], and testing set [X_test,y_test].
% Ignoring the labels y, apply PCA on X_train to get the k principal components. 
% The  principal componenents are obtained using X_train, NOT X_val or X_test.
% The k principal components define a transformation REDUC: x --> z (that can be applied to any x).
% Let Z_train=REDUC(X_train), Z_val=REDUC(X_val), Z_test=REDUC(X_test).
% Apply a supervised learning technique of interet (e.g., NN) using [Z_train,y] and [Z_val,y] to obtain a hypothesis H(z)=y.
% The we can test on [Z_test,y].
% And given a new set X for which we want to predict labels, first get Z=REDUC(X), then use labels y=H(Z).
%
% Do now use all the points available (training and validation) to construct the reducer (use only the training points). 
% Do no PCA to avoid overfitting. Use it in the data is too large given the resources.
% If possible, first try things without PCA.
