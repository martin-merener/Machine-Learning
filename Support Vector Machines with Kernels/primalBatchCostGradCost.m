function [cost, grad] = primalBatchCostGradCost(bbeta,lambda,K,Y,batch)
% A mini-batch implementation on top of the approach in 'Training a Support Vector Machine in the Primal', by Olivier Chapelle)
% NOTE: the cost is with respect to ALL training points WHILE the gradient is with respect to the batch. 
% Do this cost cannot be used to approximate the partial derivatives as (f(x+0.001)-f(x-0.001))/0.002. 
% For gradient checking, use primalBatchCostOnly_OnlyToCheckGradOfMiniBatch

b = bbeta(1);
beta = bbeta(2:end);

% cost
N = size(Y,1);
Kbeta = K*beta; % to save time
cost = (lambda*beta'*Kbeta + sum(max([zeros(N,1),1-Y.*(Kbeta+b)],[],2).^2)); 

% grad NEW ALTERNATIVE (half the time)
S1 = (Y.*(Kbeta+b)<1);
temp = zeros(N,1);
temp(batch) = 1;
S2 = temp>0;
S = S1.*S2;
grad_b = 2*S'*(Kbeta+b-Y);
S = logical(S);
T1 = K(:,S);
T2 = Kbeta(S)+b-Y(S);
grad_beta = 2*(lambda*Kbeta + T1*T2);
grad = [grad_b;grad_beta];

end

% % grad % AS BEFORE
% S = (Y.*(Kbeta+b)<1);
% grad_b = 2*S'*(Kbeta+b-Y);
% KI0 = zeros(N);
% KI0(:,S) = K(:,S); % same as K*I0 = K*diag(S);
% grad_beta = 2*(lambda*Kbeta + KI0*(Kbeta+b-Y));
% grad = [grad_b;grad_beta];
