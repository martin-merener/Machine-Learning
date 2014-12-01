% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function [cost, grad] = primalCostGradCost(bbeta,lambda,K,Y)
% computes the cost and gradient of the primal with quadratic loss (as in Training a Support Vector Machine in the Primal, by Olivier Chapelle)

b = bbeta(1);
beta = bbeta(2:end);

% cost
N = size(Y,1);
Kbeta = K*beta; % to save time
cost = (lambda*beta'*Kbeta + sum(max([zeros(N,1),1-Y.*(Kbeta+b)],[],2).^2)); 

% grad
S = (Y.*(Kbeta+b)<1);
grad_b = 2*S'*(Kbeta+b-Y);
KI0 = zeros(N);
KI0(:,S) = K(:,S); % same as K*I0 = K*diag(S);
grad_beta = 2*(lambda*Kbeta + KI0*(Kbeta+b-Y));
grad = [grad_b;grad_beta];

% NOTE: in the paper it is clear how the gradient is computed. This can be
% addapted then to other costs, including SVM for regression.

end


