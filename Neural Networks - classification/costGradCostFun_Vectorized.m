function [cost, GRAD_vect] = costGradCostFun_Vectorized(W_vect,X_train,Y_train,lambda,nClasses,D)

L = length(D);
W = vect2cell(W_vect,D);

% type of regularization:
regularizationType = 'elimination'; % 'decay' OR 'elimination'

nTrain = size(X_train,1);

% forward propagation
[X,S] = forwardProp(W,X_train);

% hypothesis
hX = X{L,1}';
costs = errorFun(hX,Y_train); % in-sample error
regTerm = 0;
for J = 1:L-1
    if strcmp(regularizationType,'decay')==1
        regTerm = regTerm + sum(sum(W{J,1}.^2)); % regularization: squared weight decay
    else
        regTerm = regTerm + sum(sum((W{J,1}.^2)./(1+W{J,1}.^2))); % regularization: weight elimination
    end
end
cost = mean(sum(costs,2)) + (lambda/nTrain)*regTerm;  

% backwards propagation (sensitivities)
if nClasses == 2
    delta = cell(L-1,1);
    [~,delta{L-1,1}] = activFun(S{L-1,1}); % derivative of activFun
    for J = L-2:-1:1
        [~,temp1] = activFun(S{J,1}); % derivative of activFun
        temp2 = W{J+1,1}*delta{J+1,1};
        temp2 = temp2(2:end,:);
        delta{J,1} = temp1.*temp2;
    end
else % for multiclass, the delta of each class is computed separately
    delta = cell(L-1,nClasses);
    [~,temp] = activFun(S{L-1,1}); % derivative of activFun
    for I = 1:nClasses
        delta{L-1,I} = temp(I,:);
    end
    for I = 1:nClasses
        for J = L-2:-1:1
            [~,temp1] = activFun(S{J,1}); % derivative of activFun
            if J+1 == L-1
                temp2 = W{J+1,1}(:,I)*delta{J+1,I};
            else
                temp2 = W{J+1,1}*delta{J+1,I};
            end
            temp2 = temp2(2:end,:);
            delta{J,I} = temp1.*temp2;
        end
    end
end
   
% gradient
if nClasses == 2
    [~, temp] = errorFun(hX,Y_train);
    grad = cell(L-1,1);
    for J = 1:L-1
        if strcmp(regularizationType,'decay')==1
            grad{J,1} = (X{J,1}.*repmat(temp',size(X{J,1},1),1))*(delta{J,1}')/nTrain + (2*lambda/nTrain)*W{J,1}; % regularization: squared weight decay 
        else
            grad{J,1} = (X{J,1}.*repmat(temp',size(X{J,1},1),1))*(delta{J,1}')/nTrain + (2*lambda/nTrain)*(W{J,1}.^2)./(1+W{J,1}.^2).^2; % regularization: weight elimination
        end
    end
    GRAD = grad;
else % for multiclass the gradients coming from the different classes are combined
    [~, temp] = errorFun(hX,Y_train);
    GRAD = cell(L-1,1);
    for J = 1:L-1
        GRAD{J,1} = zeros(size(W{J,1}));    
    end

    grad = cell(L-1,1);
    temp2 = zeros(size(GRAD{L-1,1}));
    for I = 1:nClasses
        for J = 1:L-1
            grad{J,1} = (X{J,1}.*repmat(temp(:,I)',size(X{J,1},1),1))*(delta{J,I}')/nTrain;
        end
        for J = 1:L-2
            GRAD{J,1} = GRAD{J,1}+grad{J,1};
        end
        temp2(:,I) = grad{L-1,1};
    end
    GRAD{L-1,1} = GRAD{L-1,1}+temp2;

    if strcmp(regularizationType,'decay')==1
        for J = 1:L-1
            GRAD{J,1} = GRAD{J,1} + (2*lambda/nTrain)*W{J,1}; % regularization: squared weight decay 
        end    
    else
        for J = 1:L-1
           GRAD{J,1} = GRAD{J,1} + (2*lambda/nTrain)*(W{J,1}.^2)./(1+W{J,1}.^2).^2; % regularization: weight elimination
        end
    end
end
    
GRAD_vect = cell2vect(GRAD,D);

end
