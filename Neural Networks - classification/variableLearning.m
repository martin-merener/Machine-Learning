% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function [nu,W1] = variableLearning(nu,W0,alpha,beta,X_train,Y_train,lambda,direction,mu)
% gives [nu,W1] where: in-sample(W1)<in-sample(W0)+mu, and W1=W0-nu*grad.

if mu<Inf % if mu=Inf the condition will not be satisfied, so better skip it
    while costFunOnly(updateDirection(W0,nu,direction),X_train,Y_train,lambda) >= costFunOnly(W0,X_train,Y_train,lambda) + mu
        nu = nu*beta;
    end
end
W1 = updateDirection(W0,nu,direction);
nu = nu*alpha;

end
