% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function [nu,W1] = steepestLearning_Ushape(W0,X_train,Y_train,lambda,direction)
% gives [nu,W1] where nu minimizes in-sample(W0-t*grad) with respect to t, and W1=W0-t*grad.
tic;
% first we search for a U shape configuration
nu1 = 0;
nu2 = 0.1;
trash = 0;
while costFunOnly(updateDirection(W0,nu1,direction),X_train,Y_train,lambda) <= costFunOnly(updateDirection(W0,nu2,direction),X_train,Y_train,lambda)
    nu1 = nu2;
    nu2 = 2*nu2;
    trash = trash+1;
end
nu3 = 1.5*nu2;
while costFunOnly(updateDirection(W0,nu2,direction),X_train,Y_train,lambda) >= costFunOnly(updateDirection(W0,nu3,direction),X_train,Y_train,lambda)
    nu3 = 1.5*nu3;
    trash = trash+1;
end
% now we have nu1 < nu2 < nu3, such that: in-sample(W0-nu2*grad) <= MIN(in-sample(W0-nu1*grad),in-sample(W0-nu3*grad)) 
% we say that nu1, nu2, nu3 is a U-arrangement

while abs(nu1 - nu3)>0.1
    nuHat = (nu1+nu3)/2;
    trash = trash+1;
    if nuHat<nu2
        if costFunOnly(updateDirection(W0,nu2,direction),X_train,Y_train,lambda) < costFunOnly(updateDirection(W0,nuHat,direction),X_train,Y_train,lambda) 
            nu1 = nuHat;
        else
            nu3 = nu2;
            nu2 = nuHat;
        end
    else
        if costFunOnly(updateDirection(W0,nu2,direction),X_train,Y_train,lambda) < costFunOnly(updateDirection(W0,nuHat,direction),X_train,Y_train,lambda) 
            nu3 = nuHat;
        else
            nu1 = nu2;
            nu2 = nuHat;
        end
    end
end
            
nu = nu2;

W1 = updateDirection(W0,nu,direction);
toc
end

