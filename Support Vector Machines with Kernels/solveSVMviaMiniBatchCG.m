% Martin Merener, martin.merener@gmail.com, 03-Dec-2014 %
% ------------------------------------------------------%
function [b,beta] = solveSVMviaMiniBatchCG(X,Y,lambda,kernelFun,batchFraction)
% this function will find the optimal [b,beta] that minimizes the function primalCostGradCost(bbeta,lambda,K,Y) 
% note: bbeta is just the relevant argument of the objective function here

N = size(Y,1);
K = feval(kernelFun,X,X);
bInit = rand();
betaInit = rand(N,1);
bbetaInit = [bInit;betaInit];
bbeta0 = bbetaInit; 
gradBefore = zeros(N,1);
directionBefore = zeros(N,1);
epocBound = 1000;
epoc = 0; % initialize
bestCost = Inf;
bestCosts = [10^90,10^80];

while epoc<epocBound && 1-bestCosts(2)/bestCosts(1)> 0.005

    epoc = epoc+1;
    nPerBatch = ceil(batchFraction*N);
    nBatches = ceil(N/nPerBatch);
    batches = cell(nBatches,1);
    IdsAvail = 1:N;
    for J = 1:nBatches
        IdsBatch = randsample(IdsAvail,min(length(IdsAvail),nPerBatch));
        batches{J,1} = IdsBatch;
        IdsAvail = setdiff(IdsAvail,IdsBatch);
    end
    for J = 1:nBatches % for each batch it computes
        [cost,grad] = primalBatchCostGradCost(bbeta0,lambda,K,Y,batches{J,1}); % current cost (in-sample error) and its gradient   
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
        objectiveFun = @(t)primalCostOnly(bbeta0+t*direction,lambda,K,Y); 
        options = optimoptions(@fminunc,'Algorithm','quasi-newton','Diagnostics','off','Display','off');  
        nu = fminunc(objectiveFun,1,options); % steepest nu in the gradient direction
        nu = max(10^(-12),nu); % because nu cannot be 0. THIS MINIMUM nu MAY NEED TO BE CALIBRATED
        bbeta1 = bbeta0+nu*direction; 
        bbeta0 = bbeta1;
        costs((epoc-1)*nBatches+J) = cost;
        disp(J);
    end
    bestCosts(1:end-1) = bestCosts(2:end);
    bestCosts(end) = bestCost;
    disp(epoc);
    disp(bestCost);
end
b = bbeta_final(1);
beta = bbeta_final(2:end);

end