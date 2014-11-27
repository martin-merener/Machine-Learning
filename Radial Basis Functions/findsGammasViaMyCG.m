function gammas_final = findsGammasViaMyCG(w,gammas,NXMU,Y)
% this function will find the optimal gammas that minimizes the function costGradCost(w,gammas,NXMU,Y), for the given w.

K = length(gammas);
iterBound = 1000;
iter = 0; % initialize
gammasChangeBound = 10^(-3); % upper bound
gammasChange = 1; % initialize
gammas0 = gammas; 
gradBefore = zeros(K,1);
directionBefore = zeros(K,1);
costs = zeros(iterBound,1);
bestCost = Inf;

while iter<iterBound && gammasChange>=gammasChangeBound
    iter = iter+1;
    [cost,grad] = costGradCost(w,gammas0,NXMU,Y); % current cost (in-sample error) and its gradient   
    if cost<bestCost % keep best so far in your pocket
    	bestCost = cost;
        gammas_final = gammas0;
    end
    direction = -grad;
    if norm(gradBefore)>0 
        momentum = grad'*(grad-gradBefore)/(gradBefore'*gradBefore); %Polak-Ribiere (See page 163 Scholkopf)
        direction = direction + momentum*directionBefore;
    end    
    gradBefore = grad;
    directionBefore = direction;
    objectiveFun = @(t)costOnly(w,gammas0+t*direction,NXMU,Y); 
    options = optimoptions(@fminunc,'Algorithm','quasi-newton','Diagnostics','off','Display','off');
    nu = fminunc(objectiveFun,1,options); % steepest nu in the gradient direction
    gammas1 = gammas0+nu*direction; 
    gammasChange = norm(gammas1-gammas0);
    gammas0 = gammas1; % ready for next iteration
    costs(iter) = cost;
end
costs = costs(1:iter);

end