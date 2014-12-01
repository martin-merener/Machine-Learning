% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function cost = costFunOnly(W,X_train,Y_train,lambda)

L = size(W,1)+1;
nTrain = size(X_train,1);

% forward propagation
[X,~] = forwardProp(W,X_train);

% hypothesis
hX = X{L,1}';
costs = errorFun(hX,Y_train); % in-sample error
regTerm = 0;
for J = 1:L-1
%    regTerm = regTerm + sum(sum(W{J,1}.^2)); % regularization: squared weight decay
    regTerm = regTerm + sum(sum((W{J,1}.^2)./(1+W{J,1}.^2))); % regularization: weight elimination
end
cost = mean(sum(costs,2)) + (lambda/nTrain)*regTerm;  

end

