% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function [w, iter] = perceptron_learningAlgo(X,Y,maxIter)
% Perceptron Learning Algorithm
%(X,Y) form the training set (points and labels).
% maxIter is an upper bound on the number of iterations, in case the set is
% not linearly separable.

[n,d] = size(X);

w = zeros(d+1,1);

X_e = [ones(n,1), X];

Y_est = zeros(size(Y));
Y_est(X_e*w>=0) = 1;
Y_est(X_e*w<0) = -1;

iter = 0;

while ~isequal(Y_est,Y) && iter <= maxIter
    iter = iter+1;
    k = find(Y_est~=Y,1,'first');
    if ~isempty(k) 
        w = w+Y(k)*X_e(k,:)';
    end
    Y_est = zeros(size(Y));
    Y_est(X_e*w>=0) = 1;
    Y_est(X_e*w<0) = -1;
end

end

