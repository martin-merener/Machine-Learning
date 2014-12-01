% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%

function errors = kNN_learning(X, Y, Ks, estimationType)
% This function gives the optimal k for kNN. It is based on leave-one-out applied to the training set.

nKs = length(Ks);
errors = zeros(nKs,2);

m = size(X,1); 
m_ref = m-1;

for k = 1:nKs
    K = Ks(k);
    Y_est_k = zeros(m,1);
    for l = 1:m
        current_x = X(l,:);
        ref_pts = X([(1:l-1),(l+1:m)]',:);
        ref_labels = Y([(1:l-1),(l+1:m)]',:);
        distancesTo_x = sqrt(sum((ref_pts-repmat(current_x,m_ref,1)).^2,2));
        sortedDistances = sort(distancesTo_x);
        ids = distancesTo_x <= sortedDistances(K);
        if strcmp(estimationType,'classification')
            Y_est_k(l) = mode(ref_labels(ids));
        else
            Y_est_k(l) = mean(ref_labels(ids));
        end
    end
    errors(k,:) = [K, mean(abs(Y_est_k-Y))];
end

end

