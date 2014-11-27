function [nu,W1] = steepestLearning_Vectorized(W0,X_train,Y_train,lambda,direction,D)
% gives [nu,W1] where nu minimizes in-sample(W0-t*grad) with respect to t, and W1=W0-t*grad.

options = optimset('Diagnostics','off','Display','off');
objFun = @(t) costFunOnly_Vectorized(updateDirection_Vectorized(W0,t,direction),X_train,Y_train,lambda,D);
nu = fminunc(objFun,1,options); % steepest nu in the gradient direction
W1 = W0 + nu*direction;
 
end

