% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function [bestCost,bestIdx,bestCentroids] = kMeansClustering(X,K)
%Given the training points, and the number of clusters, it returns the
%cost, the centroids, and the clusters given by the best initialization

N = size(X,1);
costChangeBound = 0.0001; % cost change that would make the search stop
iterBound = 1000; % bound on number of iterations
nRuns = 100; % each run with a different initialization of the centroids
bestCost = Inf;

for J = 1:nRuns
    
    costChange = costChangeBound+1; % initializing
    iter = 0; % initializing
    cost = Inf; % initializing
    centroids = X(randsample(N,K),:); % centroid initialization

    while costChange>costChangeBound && iter<iterBound
        iter = iter+1;
        costBefore = cost;
        idx = findClosestCentroids(X,centroids);
        centroids = computeCentroids(X,idx,K);
        cost = distortionCost(X,idx,centroids,K);
        costChange = costBefore-cost;
    end

    if cost < bestCost
        bestCost = cost;
        bestIdx = idx;
        bestCentroids = centroids;
    end
    
end

% gscatter(X(:,1),X(:,2),bestIdx,'bgrcmyk','o',2);      
% axis([0 1 0 1])
% title('Training X with obtained clustering');   

end

