function [gammas, w] = learnRBFviaMyCG(NXMU,Y)
%Given the centers, and the labeled data points, this function returns the
%gammas for the RBF of choice (specified in radialOnNorms). It is assumed
%to be used for regression or binary classification. 

K = size(NXMU,2);
maxIter = 2000;
iter = 0;
costBefore = Inf;
costChangeBound = 0.0001;
costChange = costChangeBound+1;
costs = zeros(maxIter,1);
gammas = 100*rand(K,1); % initializing gammas
while iter<maxIter && costChange>=costChangeBound
    iter = iter+1;
    disp(iter);
    % for the current gammas, we compute the optimal weights w
    PHI = radialOnNorms(NXMU,gammas);
    w = pinv(PHI'*PHI)*PHI'*Y; % with least squares the optimal weights are w = pinv(Phi'*Phi)*Phi'*y. Note that we bias term "b" is desired, just add a column of ones on the left of PHI
    % for the current w, we compute the optimal gammas using CG
    [gammas] = findsGammasViaMyCG(w,gammas,NXMU,Y);
    [cost,~] = costGradCost(w,gammas,NXMU,Y);
    costs(iter) = cost;
    costChange = abs(cost-costBefore);
    costBefore = cost;
end

end

