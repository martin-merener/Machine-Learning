% Script implementing 'Training a Support Vector Machine in the Primal', by Olivier Chapelle
% Binary and Multiclass, non-linearly separable classes, via Kernels and Newton method (uses gradient & hessian).

clear all

% -------------------- INITIALIZING ---------------------------------------
% -------------------- GENERATING DATA ------------------------------------
%
d = 2; % dimension of space
N = 3000; % number of training points           
N_test = 1000; % number of testing points
nClasses = 3; % number of classes
nC = 0; % actual number of classes in training set
while nC < nClasses % generates training set until nC=nClasses
    [X_data,Y_data,trueCenters] = generatingPoints4Classification(N+N_test,d,nClasses);
    idx_train = randsample(N+N_test,N);
    idx_test = setdiff(1:N+N_test,idx_train);
    X_train = X_data(idx_train,:);
    X_test = X_data(idx_test,:);
    Y_train = Y_data(idx_train,:);
    Y_test = Y_data(idx_test,:);
    classes = unique(Y_train);
    nC = length(classes);
end


% -------------------- SVM ALGORITHM --------------------------------------
% -------------------- LEARNING b and beta --------------------------------
%
kernelFun = 'gauss_kXY';
X = X_train;
lambda = 0.1;
if nClasses == 2 % binary
	Y = 2*(Y_train == classes(1))-1; % Classes are +1,-1
    [b,beta] = solveSVMviaNewton(X,Y,lambda,kernelFun);
else % multiclass, via one-vs-all method
    B = zeros(nClasses,1);
    BETA = zeros(size(X,1),nClasses);
    for J = 1:nClasses
        Y = 2*(Y_train == classes(J))-1; % Classes are +1,-1
        [b,beta] = solveSVMviaNewton(X,Y,lambda,kernelFun);      
        B(J) = b;
        BETA(:,J) = beta;
    end
end


% -------------------- ESTIMATION ON --------------------------------------
% -------------------- TRAINING SET ------------------------------------------
%
X = X_train;
if nClasses == 2 % binary
    hX = feval(kernelFun,X,X_train)*beta + b;
    Y_train_estimation = (sign(hX)==1)*classes(1)+(sign(hX)==-1)*classes(2);
    inSampleE = 1 - mean(Y_train == Y_train_estimation);
else % multiclass, via one-vs-all method
    HX = zeros(size(X,1),nClasses);
    for J = 1:nClasses
        hX = feval(kernelFun,X,X_train)*BETA(:,J) + B(J);
        HX(:,J) = hX;
    end
    [~,idxClasses] = max(HX,[],2);
    Y_train_estimation = zeros(size(X,1),1);
    for I = 1:size(X,1)
        Y_train_estimation(I) = classes(idxClasses(I));
    end
    inSampleE = 1-mean(Y_train_estimation==Y_train);
end


% -------------------- ESTIMATION ON --------------------------------------
% -------------------- TESTING SET ------------------------------------------
%
X = X_test;
if nClasses == 2 % binary
    hX = feval(kernelFun,X,X_train)*beta + b;
    Y_test_estimation = (sign(hX)==1)*classes(1)+(sign(hX)==-1)*classes(2);
    outSampleE = 1 - mean(Y_test == Y_test_estimation);
else % multiclass, via one-vs-all method
    HX = zeros(size(X,1),nClasses);
    for J = 1:nClasses
        hX = feval(kernelFun,X,X_train)*BETA(:,J) + B(J);
        HX(:,J) = hX;
    end
    [~,idxClasses] = max(HX,[],2);
    Y_test_estimation = zeros(size(X,1),1);
    for I = 1:size(X,1)
        Y_test_estimation(I) = classes(idxClasses(I));
    end
    outSampleE = 1-mean(Y_test_estimation==Y_test);
end


% -------------------- DISPLAYING RESULTS ON ------------------------------
% -------------------- TRAINING AND TESTING -------------------------------
%
if d==2
    
    figure                                                       
    subplot(2,3,1);
    gscatter(X_train(:,1),X_train(:,2),Y_train,'bgrcmyk','o',3);      
    axis([0 1 0 1]);
    title('Training X vs Training Labels Y');                                       

    subplot(2,3,2);
    gscatter(X_train(:,1),X_train(:,2),Y_train_estimation,'bgrcmyk','o',3);      
    axis([0 1 0 1]);
    title('Training X vs Estimated Labels Y');                                       

    subplot(2,3,3);
    gscatter(X_train(:,1),X_train(:,2),Y_train_estimation~=Y_train,'cr','o',3);      
    axis([0 1 0 1]);
    title(strcat('Training X vs Correctness. In Sample Error: ',num2str(100*inSampleE),'%'));                                       
    
    subplot(2,3,4);
    gscatter(X_test(:,1),X_test(:,2),Y_test,'bgrcmyk','o',3);      
    axis([0 1 0 1]);
    title('Testing X vs Testing Labels Y');                                       

    subplot(2,3,5);
    gscatter(X_test(:,1),X_test(:,2),Y_test_estimation,'bgrcmyk','o',3);      
    axis([0 1 0 1]);
    title('Testing X vs Estimated Labels Y');                                       

    subplot(2,3,6);
    gscatter(X_test(:,1),X_test(:,2),Y_test_estimation~=Y_test,'cr','o',3);      
    axis([0 1 0 1]);
    title(strcat('Testing X vs Correctness. Out of Sample Error: ',num2str(100*outSampleE),'%'));                                       
end



% bbeta = bbetaInit;
% [cost, grad] = primalCostGradCost(bbeta,lambda,K,Y);
% 
% % % Gradient checking
% epsilon = 0.0001;
% gradApprox = zeros(length(bbeta),1);
% for J = 1:length(bbeta)
%     disp(J);
%     betabPlus = bbeta;
%     betabPlus(J) = betabPlus(J)+epsilon;
%     betabMinus = bbeta;
%     betabMinus(J) = betabMinus(J)-epsilon;
%     gradApprox(J) = (primalCostGradCost(betabPlus,lambda,K,Y) - primalCostGradCost(betabMinus,lambda,K,Y))/(2*epsilon);
% end
% scatter(gradApprox,grad);
