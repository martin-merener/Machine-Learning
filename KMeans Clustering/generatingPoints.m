% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function [X,trueIds,trueCentroids] = generatingPoints(m,d,K)
% It creates a set of points, that can be used for training, validation or testing.
% It creates m points in [0,1]^d. To generate the set of points, if first
% creates p centroids at random, and then it generates points around each
% centroid.

r = 0.4; % radius of influence of each center

trueCentroids = rand(K,d); % the true centers that determine the classes
trueIdxCenters = (1:K); % indicates the cluster to which each center belongs

% number of points per center
m_center = zeros(K,1); 
m_center(1:K-1) = floor(m/K);
m_center(K) = m-sum(m_center);

X = zeros(m,d);
trueIds = zeros(m,1);

rowUpto = 0;
for J = 1:K
    temp = 2*rand(m_center(J),d)-1; % random points in [-1,1]^d
    norms_temp1 = sqrt(sum(temp.^2,2)); % norms of those random points
    norms_temp2 = 1./abs(normrnd(0,r,m_center(J),1)); % to obtain random norms in (0,r)
    temp = r*temp./repmat(norms_temp1.*norms_temp2,1,d); % random points in the circle of radius r (the radius are distributed as abs(norm)
    temp = repmat(trueCentroids(J,:),m_center(J),1)+temp; % random points in the circle or radius r added to the current center
    X(rowUpto+1:rowUpto+m_center(J),:) = temp; % storing training points
    trueIds(rowUpto+1:rowUpto+m_center(J),:) = trueIdxCenters(J); % storing classes of training points
    rowUpto = rowUpto+m_center(J); 
end

end

