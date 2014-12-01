% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function [nu,W1] = steepestLearning(W0,X_train,Y_train,lambda,direction)
% gives [nu,W1] where nu minimizes in-sample(W0-t*grad) with respect to t, and W1=W0-t*grad.

options = optimset('Diagnostics','off','Display','off');
objFun = @(t) costFunOnly(updateDirection(W0,t,direction),X_train,Y_train,lambda);
nu = fminunc(objFun,1,options); % steepest nu in the gradient direction
W1 = updateDirection(W0,nu,direction);
 
end

