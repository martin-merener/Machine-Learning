% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function [best_w, iter] = perceptronPocketBetterInitial_learningAlgo(X,Y,maxIter)
% Perceptron Pocket Learning Algorithm. Pocket meaning that it keeps the
% best solution so far with respect to the in-sample error.
%(X,Y) form the training set (points and labels).
% maxIter is an upper bound on the number of iterations, in case the set is
% not linearly separable.

n = size(X,1);

X_e = [ones(n,1), X];

w = ((X_e'*X_e)\(X_e'))*Y;
best_w = w;
lowest_error = 1;

Y_est = zeros(size(Y));
Y_est(X_e*w>=0) = 1;
Y_est(X_e*w<0) = -1;

iter = 0;

while ~isequal(Y_est,Y) && iter <= maxIter
    iter = iter+1;
%    misses = find(Y_est~=Y);           % either a random point among the misses.
%     if length(misses)==1
%         k = misses;
%     else
%         k = randsample(misses,1);
%     end
    k = find(Y_est~=Y,1,'first');       % or the first point missed.
    if ~isempty(k) 
        w = w+Y(k)*X_e(k,:)';
    end
    Y_est = zeros(size(Y));
    Y_est(X_e*w>=0) = 1;
    Y_est(X_e*w<0) = -1;
    current_error = mean(abs(Y_est-Y));
    if current_error<=lowest_error
        lowest_error = current_error;
        best_w = w;
    end
end

end

