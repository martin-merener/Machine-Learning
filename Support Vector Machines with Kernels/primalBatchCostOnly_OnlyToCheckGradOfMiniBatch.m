% Martin Merener, martin.merener@gmail.com, 03-Dec-2014 %
% ------------------------------------------------------%

function cost = primalBatchCostOnly_OnlyToCheckGradOfMiniBatch(bbeta,lambda,K,Y,batch)
% computes the cost with respect to the BATCH. ONLY USED TO CHECK THE GRADIENT OF THE MINI-BATCH implementation

b = bbeta(1);
beta = bbeta(2:end);

% cost
N = size(Y,1);
temp = zeros(N,1);
temp(batch) = 1;
S = temp>0;
Kbeta = K*beta; % to save time
cost = (lambda*beta'*Kbeta + sum(S.*max([zeros(N,1),1-Y.*(Kbeta+b)],[],2).^2)); 

end

% MINI BATCH GRADIENT CHECKING CODE
% [cost,grad] = primalBatchCostGradCost(bbeta0,lambda,K,Y,batches{1,1});
% gradApprox = zeros(size(grad));
% epsilon = 0.0001;
% for L = 1:length(grad)
%     bbeta0Plus = bbeta0;
%     bbeta0Plus(L) = bbeta0Plus(L)+epsilon;
%     bbeta0Minus = bbeta0;
%     bbeta0Minus(L) = bbeta0Minus(L)-epsilon;
%     gradApprox(L) = (primalBatchCostOnly_NOTUSED(bbeta0Plus,lambda,K,Y,batches{1,1})-primalBatchCostOnly_NOTUSED(bbeta0Minus,lambda,K,Y,batches{1,1}))/(2*epsilon);
% end