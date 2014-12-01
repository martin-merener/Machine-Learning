% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function cost = primalCostOnly(bbeta,lambda,K,Y)
% computes the cost of the primal with quadratic loss (as in Training a Support Vector Machine in the Primal, by Olivier Chapelle)

b = bbeta(1);
beta = bbeta(2:end);

% cost
N = size(Y,1);
Kbeta = K*beta; % to save time
cost = (lambda*beta'*Kbeta + sum(max([zeros(N,1),1-Y.*(Kbeta+b)],[],2).^2)); 

end


