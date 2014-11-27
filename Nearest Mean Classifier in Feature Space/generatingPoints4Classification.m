function [X_data,Y_data,trueCenters] = generatingPoints4Classification(N,d,C)
% It creates a set of points, that can be used for training, validation or testing.
% It creates N points in [0,1]^d. To generate the set of points, if first
% creates p=factor*C centers at random, and divides them into C classes. 
% Then it generataes points at random in a neibhborhood of each center.
% Each center determines a class.

factor = 2; % this indicates how many centers each class will have
p = factor*C; % total number of centers
r = 0.3; % radius of influence of each center

trueCenters = rand(p,d); % the true centers that determine the classes
trueIdxCenters = ceil((1:p)/factor); % indicates the cluster to which each center belongs

% number of points per center
N_center = zeros(p,1); 
N_center(1:p-1) = floor(N/p);
N_center(p) = N-sum(N_center);

X_data = zeros(N,d);
Y_data = zeros(N,1);

rowUpto = 0;
for J = 1:p
    temp = 2*rand(N_center(J),d)-1; % random points in [-1,1]^d
    norms_temp1 = sqrt(sum(temp.^2,2)); % norms of those random points
    norms_temp2 = 1./abs(normrnd(0,r,N_center(J),1)); % to obtain random norms in (0,r)
    temp = r*temp./repmat(norms_temp1.*norms_temp2,1,d); % random points in the circle of radius r (the radius are distributed as abs(norm)
    temp = repmat(trueCenters(J,:),N_center(J),1)+temp; % random points in the circle or radius r added to the current center
    X_data(rowUpto+1:rowUpto+N_center(J),:) = temp; % storing training points
    Y_data(rowUpto+1:rowUpto+N_center(J),:) = trueIdxCenters(J); % storing classes of training points
    rowUpto = rowUpto+N_center(J); 
end

end

