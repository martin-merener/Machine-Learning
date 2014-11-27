function [X,S] = forwardProp(W,X_train)

L = size(W,1)+1;
nTrain = size(X_train,1);

X = cell(L,1); % the values of x on each layer
S = cell(L-1,1); % the signals on each layer (except input layer 1)
X{1,1} = [ones(1,nTrain); X_train'];
for J = 2:L
    S{J-1,1} = (W{J-1,1})'*X{J-1,1};
    if J < L % this conditional is because in the output layer we don't add bias units
        X{J,1} = [ones(1,nTrain); activFun(S{J-1,1})];
    else
        X{J,1} = activFun(S{J-1,1});
    end
end

end