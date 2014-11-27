function centroids = computeCentroids(X,idx,K)
%returs the new centroids by computing the means of the data points assigned to each centroid.

d = size(X,2); % dimension
centroids = zeros(K,d); % new centroids
for J = 1:K % loop recomputing new centroids
    centroids(J,:) = mean(X(idx==J,:),1);
end

end

