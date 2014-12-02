% Martin Merener, martin.merener@gmail.com, 02-Dec-2014 %
% ------------------------------------------------------%
function [X,Theta,costs] = solveCollFilt_viaCG(Y, R, N_features, lambda)
% This function will find the optimal [X,Theta] that minimizes the function [J, grad] = costGradCost(XTheta, Y, N_features, lambda)

[N_items,N_users] = size(Y);
X = 0.02*rand(N_items,N_features)-0.01; 
Theta = 0.02*rand(N_users,N_features)-0.01; 
XTheta = [X(:);Theta(:)];

N = size(XTheta,1);
iterBound = 2000;
iter = 0; % initialize
coeffsChangeBound = 0.05; % upper bound
coeffsChanges = ones(20,1); % initialize
coeffs0 = XTheta; 
gradBefore = zeros(N,1);
directionBefore = zeros(N,1);
costs = zeros(iterBound,1);
bestCost = Inf;

while iter<iterBound && max(coeffsChanges)>=coeffsChangeBound
    iter = iter+1;
    [cost,grad] = costGradCost(coeffs0, Y, R, N_features, lambda); % current cost (in-sample error) and its gradient   
    if cost<bestCost % keep best so far in your pocket
    	bestCost = cost;
        coeffs_final = coeffs0;
    end
    direction = -grad;
    if norm(gradBefore)>0 
        momentum = grad'*(grad-gradBefore)/(gradBefore'*gradBefore); %Polak-Ribiere (See page 163 Scholkopf)
        direction = direction + momentum*directionBefore;
    end    
    gradBefore = grad;
    directionBefore = direction;
    objectiveFun = @(t)costOnly(coeffs0+t*direction, Y, R, N_features, lambda);
    options = optimoptions(@fminunc,'Algorithm','quasi-newton','Diagnostics','off','Display','off');  
    nu = fminunc(objectiveFun,1,options); % steepest nu in the gradient direction
    nu = max(10^(-9),nu); % because nu cannot be 0. THIS MINIMUM nu MAY NEED TO BE CALIBRATED
    coeffs1 = coeffs0+nu*direction; 
    coeffsChange = norm(coeffs1-coeffs0); 
    coeffsChanges(1:end-1) = coeffsChanges(2:end);
    coeffsChanges(end) = coeffsChange;
    coeffs0 = coeffs1; % ready for next iteration
    costs(iter) = cost;
    if rem(iter,100)==0
        disp([iter/100, cost, bestCost, coeffsChange, max(coeffsChanges), nu]);
    end
end
disp([iter/100, cost, bestCost, coeffsChange, max(coeffsChanges), nu]);
costs = costs(1:iter);
[N_items,N_users] = size(Y);
X = reshape(coeffs_final(1:N_items*N_features), N_items, N_features);
Theta = reshape(coeffs_final(N_items*N_features+1:end), N_users, N_features);

end