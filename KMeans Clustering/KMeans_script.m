% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%

% K-MEANS CLUSTERING. 
% 1) Initialize centroids
% 2) Iterate:
% 2.1) Compute clusters (assign points to closest centroid) 
% 2.2) Compute centroids (compute centers of each cluster) 
%
% Could also be: 
% 1) Initialize clusters
% 2) Iterate:
% 2.1) Compute centroids (compute centers of each cluster) 
% 2.2) Compute clusters (assign points to closest centroid) 
%
% This algorithm actually minimizes the "distortion" cost function:
% 
% J(c^(1),...,c^(m),mu_1,...,mu_K)) = sum_{i=1}^m || x^(i) - mu_c(i) || 
% 
% where m is the number of training points, 
% c^(i) is the cluster of the training point x^(i), 
% and mu_j is the centroid of the j-th cluster     
%
% Pseudocode (again):
% 1) Initialize K cluster centroids mu_1,...,mu_K, as K training points chosen at random
% 2) Iterate: 
% 2.1) compute c^(1),...,c^(m)
% 2.2) compute mu_1,...,mu_K
%
% Try multiply random initializations. For example 100 times. Pick the clustering with the lowest cost.
% To choose the number of cluster (by hand). The more clusters, the lower the error. But sometimes we can see an “elbow”, a point where the cost changes its decreasing rate (lower decreasing rate). But sometimes the elbow is not clear at all in practice. Sometimes the number of clusters is determined by a later criteria related to the purpose of the cluster analysis.

clear all

% Building points
m = 1000;
d = 2; % space dimension
trueK = 4;
[X,trueIds,trueCentroids] = generatingPoints(m,d,trueK);

% Feature normalization
% X = X - repmat(mean(X,1),m,1);
% X = X./repmat(std(X),m,1);

% K-Means
K = 6;
costChangeBound = 0.0001; % cost change that would make the search stop
iterBound = 1000; % bound on number of iterations

nRuns = 100;
bestCost = Inf;

for J = 1:nRuns

    disp(J);
    
    costChange = costChangeBound+1; % initializing
    iter = 0; % initializing
    costNow = Inf; % initializing
    centroids = X(randsample(m,K),:); % centroid initialization

    while costChange>costChangeBound && iter<iterBound
        iter = iter+1;
        idx = findClosestCentroids(X,centroids);
        centroids = computeCentroids(X,idx,K);
        costBefore = costNow;
        costNow = distortionCost(X,idx,centroids,K);
        costChange = costBefore-costNow;
    end

    if costNow < bestCost
        bestCost = costNow;
        bestIdx = idx;
    end
    
end

figure                                                       
subplot(1,2,1);
gscatter(X(:,1),X(:,2),trueIds,'bgrcmyk','o',2);     
axis([0 1 0 1])
title('Training X with original clustering');                                       

subplot(1,2,2);
gscatter(X(:,1),X(:,2),bestIdx,'bgrcmyk','o',2);      
axis([0 1 0 1])
title('Training X with obtrained clustering');                                       

disp(bestCost);
