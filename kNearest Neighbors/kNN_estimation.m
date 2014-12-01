% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%

function Y_est = kNN_estimation(X_ref, Y_ref, X, K, estimationType)
% Computing the estimation based on the optimal K.

m = size(X,1); 
m_ref = size(X_ref,1); 
Y_est = zeros(m,1);

for l = 1:m
    current_x = X(l,:);
    ref_pts = X_ref;
    ref_labels = Y_ref;
    distancesTo_x = sqrt(sum((ref_pts-repmat(current_x,m_ref,1)).^2,2));
    sortedDistances = sort(distancesTo_x);
    ids = distancesTo_x <= sortedDistances(K);
    if strcmp(estimationType,'classification')
    	Y_est(l) = mode(ref_labels(ids));
    else
    	Y_est(l) = mean(ref_labels(ids));
    end
end
        
end