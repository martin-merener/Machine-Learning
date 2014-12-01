% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function [b,beta] = solveSVMviaMyCG(X,Y,lambda,kernelFun)
% this function will find the optimal [b,beta] that minimizes the function primalCostGradCost(bbeta,lambda,K,Y) 
% note: bbeta is just the relevant argument of the objective function here

N = size(Y,1);
K = feval(kernelFun,X,X);
bInit = rand();
betaInit = rand(N,1);
bbetaInit = [bInit;betaInit];
iterBound = 500;
iter = 0; % initialize
bbetaChangeBound = 10^(-2); % upper bound
bbetaChange = 1; % initialize
bbeta0 = bbetaInit; 
gradBefore = zeros(N,1);
directionBefore = zeros(N,1);
costs = zeros(iterBound,1);
bestCost = Inf;

while iter<iterBound && bbetaChange>=bbetaChangeBound
    iter = iter+1;
    [cost,grad] = primalCostGradCost(bbeta0,lambda,K,Y) ; % current cost (in-sample error) and its gradient   
    if cost<bestCost % keep best so far in your pocket
    	bestCost = cost;
        bbeta_final = bbeta0;
    end
    direction = -grad;
    if norm(gradBefore)>0 
        momentum = grad'*(grad-gradBefore)/(gradBefore'*gradBefore); %Polak-Ribiere (See page 163 Scholkopf)
        direction = direction + momentum*directionBefore;
    end    
    gradBefore = grad;
    directionBefore = direction;
    %objectiveFun = @(t)objectiveFunMYCG(t,bbeta0,direction,lambda,K,Y);
    objectiveFun = @(t)primalCostOnly(bbeta0+t*direction,lambda,K,Y); 
    options = optimoptions(@fminunc,'Algorithm','quasi-newton','Diagnostics','off','Display','off');  
    nu = fminunc(objectiveFun,1,options); % steepest nu in the gradient direction
    bbeta1 = bbeta0+nu*direction; 
    bbetaChange = norm(bbeta1-bbeta0);
    bbeta0 = bbeta1; % ready for next iteration
    costs(iter) = cost;
end
costs = costs(1:iter);
b = bbeta_final(1);
beta = bbeta_final(2:end);

end