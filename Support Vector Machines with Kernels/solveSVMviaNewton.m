% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function [b,beta] = solveSVMviaNewton(X,Y,lambda,kernelFun)
% Solve SVM via Kernels, using Newton approach as explained in 'Training a Support Vector Machine in the Primal', by Olivier Chapelle

N = size(Y,1);
K = feval(kernelFun,X,X); % % relation #pts vs seconds: (40000, 22), (30000,12), (20000,5), (10000,2)
bInit = rand();
betaInit = rand(N,1);
bbetaInit = [bInit;betaInit];
bbeta = bbetaInit;
bbetaChangeBound = 0.00001;
bbetaChange = Inf;
Idx_record = 1:size(X,1);
while bbetaChange>bbetaChangeBound
    disp(bbetaChange);
    b = bbeta(1);
    beta = bbeta(2:end);
    sv = find(Y.*(K*beta+b)<1);
    notsv = setdiff(1:size(X,1),sv)';
    Ksv = K(sv,sv);
    Isv = eye(length(sv));
    Ysv = Y(sv);
    betabBefore = [b;beta([sv;notsv])];
    temp1 = [0,ones(1,length(sv))];
    temp2 = [ones(length(sv),1),lambda*Isv+Ksv];
    bbeta = pinv([temp1;temp2])*[0;Ysv]; 
    b = bbeta(1);
    beta = [bbeta(2:end);zeros(length(notsv),1)];
    bbeta = [b;beta];
    X = X([sv;notsv]);
    Y = Y([sv;notsv]);
    K = K([sv;notsv],[sv;notsv]');
    bbetaChange = norm(bbeta-betabBefore);
    Idx_record = Idx_record([sv;notsv]);
end
[~,originalOrder] = sort(Idx_record);
beta = beta(originalOrder);


% sv = find(Y.*(K*beta)<1);
% notsv = setdiff(1:size(X,1),sv)';
% beta = beta([sv;notsv]);
% X = X([sv;notsv]);
% Y = Y([sv;notsv]);
% K = K([sv;notsv],[sv;notsv]');
% Idx_record = Idx_record([sv;notsv]);


end

