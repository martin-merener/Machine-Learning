% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%

% Using Linear Regression with Regularization for Classification (one-vs-all)
clear all


% -------------------- INITIALIZATION -------------------------------------
% --------------------                -------------------------------------
%
% It is assumed that there is trainSet in memory, with labels in 1st column and features in the other columns
% It is assumed that there is testSet in memory, with labels in 1st column and features in the other columns

% Regularizer
lambda = 1;

% Train data
X_train = trainSet(:,2:end);
Y_train = trainSet(:,1);
classes = unique(Y_train);
nClasses = length(classes);
%X_train = [ones(size(X_train,1),1),X_train];
X_train = [ones(size(X_train,1),1),X_train,X_train(:,1).*X_train(:,2),X_train(:,1).^2,X_train(:,2).^2];
d = size(X_train,2);


% -------------------- LINEAR CLASSIFICATION ------------------------------
% -------------------- LEARNING w's          ------------------------------
%
% one versus all
W = zeros(d,nClasses);
X = X_train;
inSampleE = zeros(nClasses,1);
for J = 1:nClasses
    Y = 2*(Y_train == classes(J))-1; % Classes are +1,-1
    w = ((X'*X + lambda*eye(size(X,2)))\(X'))*Y;
    W(:,J) = w;
    inSampleE(J) = 1-mean(sign(X*w)==Y);
end


% -------------------- ESTIMATION ON --------------------------------------
% -------------------- TESTING SET ------------------------------------------
%
% Test data
X_test = testSet(:,2:3);
Y_test = testSet(:,1);
%X_test = [ones(size(X_test,1),1),X_test];
X_test = [ones(size(X_test,1),1),X_test,X_test(:,1).*X_test(:,2),X_test(:,1).^2,X_test(:,2).^2];
X = X_test;
outSampleE = zeros(nClasses,1);

for J = 1:nClasses
    Y = 2*(Y_test == classes(J))-1; % Classes are +1,-1
    outSampleE(J) = 1-mean(sign(X*W(:,J))==Y);
end

trash = [inSampleE,outSampleE];
disp(trash);
