% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function [alpha,b] = solveSVMviaQP(X,Y,C,kernelFun)
% Solve SVM via Kernels, using QP

MaxIter = 1000; % Matlab default is 200
N = size(X,1);
H = (Y*Y').*feval(kernelFun,X,X);
f = -ones(N,1);
Aeq = Y';
beq = 0;
lb = zeros(N,1);
ub = repmat(C,N,1);

% Optimization:
options = optimset;
options = optimset(options,'Display', 'off');
options = optimset(options,'Algorithm', 'interior-point-convex');
options = optimset(options,'MaxIter', MaxIter);
[alpha,fval,exitflag,output,lambda] = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);

% Coefficients for the hypothesis:
S = intersect(find(alpha>0.0001),find(alpha<C)); % support vectors
b = mean(Y(S)) - mean(feval(kernelFun,X(S,:),X(S,:))*(Y(S).*alpha(S)));

end

