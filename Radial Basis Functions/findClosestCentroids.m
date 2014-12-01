% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function idx = findClosestCentroids(X, centroids)
%computes the centroid memberships for every example
%idx gives the closest centroid to each point in X

m = size(X,1); % number of points
K = size(centroids,1); % number of centroids

distsqXtoCentroids = zeros(m,K); % in each row i, the distance (squared) from training point i to each of the centroids
for J = 1:K
    distsqXtoCentroids(:,J) = sum((X-repmat(centroids(J,:),m,1)).^2,2);
end
    
[~,idx] = min(distsqXtoCentroids,[],2);

end
