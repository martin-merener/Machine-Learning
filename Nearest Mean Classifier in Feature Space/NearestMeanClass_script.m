% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%

% Script for the algorithm: Nearest Mean Classifier in a Feature Space.
% Explained in Learning with Kernels, Scholkopf-Smola

clear all

% -------------------- INITIALIZING ---------------------------------------
% -------------------- GENERATING DATA ------------------------------------
%
d = 2; % dimension of space
N = 500; % number of training points           
N_test = 1000; % number of testing points
C = 4; % number of classes
nClasses = 0; % actual number of classes in training set
while nClasses < C % generates training set until nClasses=C
    [X_data,Y_data,trueCenters] = generatingPoints4Classification(N+N_test,d,C);
    idx_train = randsample(N+N_test,N);
    idx_test = setdiff(1:N+N_test,idx_train);
    X_train = X_data(idx_train,:);
    X_test = X_data(idx_test,:);
    Y_train = Y_data(idx_train,:);
    Y_test = Y_data(idx_test,:);
    classes = unique(Y_train);
    nClasses = length(classes);
end
    
% % Previous data generator for binary classification (points uniformly distributed + non-linear boundary 
% d = 2;
% n_train = 100;
% X_train = rand(n_train,d);
% boundary = @(t)(1.5*t.^2.*abs(sin(8*t)/2)+0.2);
% Y_train = 2*(boundary(X_train(:,1))>X_train(:,2))-1;
% n_test = 5000;
% X_test = rand(n_test,d);
% Y_test = 2*(boundary(X_test(:,1))>X_test(:,2))-1;


% -------------------- NEAREST MEAN ALGORITHM -----------------------------
% -------------------- LEARNING b's -------------------
%
kernelFun = 'gauss_kXY';

if nClasses == 2 % binary
	Y_temp = 2*(Y_train == classes(1))-1; % Classes are +1,-1
    X_trainPs = X_train(Y_temp==+1,:);
    X_trainNg = X_train(Y_temp==-1,:);
    K_trainPs = feval(kernelFun,X_trainPs,X_trainPs); 
    K_trainNg = feval(kernelFun,X_trainNg,X_trainNg); 
    b = (1/2)*(mean(K_trainPs(:)) - mean(K_trainNg(:)));
else % multiclass, via one-vs-all method
    X_trainPs_cell = cell(nClasses,1);
    X_trainNg_cell = cell(nClasses,1);
    B = zeros(nClasses,1);
    for J = 1:nClasses
        Y_temp = 2*(Y_train == classes(J))-1; % Classes are +1,-1
        X_trainPs = X_train(Y_temp==+1,:);
        X_trainNg = X_train(Y_temp==-1,:);
        K_trainPs = feval(kernelFun,X_trainPs,X_trainPs); 
        K_trainNg = feval(kernelFun,X_trainNg,X_trainNg); 
        b = (1/2)*(mean(K_trainPs(:)) - mean(K_trainNg(:)));
        X_trainPs_cell{J,1} = X_trainPs;
        X_trainNg_cell{J,1} = X_trainNg;
        B(J) = b;
    end
end


% -------------------- ESTIMATION ON --------------------------------------
% -------------------- TRAINING SET ------------------------------------------
%
X = X_train;
if nClasses == 2 % binary
    K_X_trainPs = feval(kernelFun,X,X_trainPs); 
    K_X_trainNg = feval(kernelFun,X,X_trainNg); 
    hX = mean(K_X_trainPs,2) - mean(K_X_trainNg,2) + b;
    Y_train_estimation = (sign(hX)==1)*classes(1)+(sign(hX)==-1)*classes(2);
    inSampleE = 1 - mean(Y_train == Y_train_estimation);
else % multiclass, via one-vs-all method
    HX = zeros(size(X,1),nClasses);
    for J = 1:nClasses
        K_X_trainPs = feval(kernelFun,X,X_trainPs_cell{J,1}); 
        K_X_trainNg = feval(kernelFun,X,X_trainNg_cell{J,1}); 
        hX = mean(K_X_trainPs,2) - mean(K_X_trainNg,2) + B(J);
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
% -------------------- TESTING SET ----------------------------------------
%
X = X_test;
if nClasses == 2 % binary
    K_X_trainPs = feval(kernelFun,X,X_trainPs); 
    K_X_trainNg = feval(kernelFun,X,X_trainNg); 
    hX = mean(K_X_trainPs,2) - mean(K_X_trainNg,2) + b;
    Y_test_estimation = (sign(hX)==1)*classes(1)+(sign(hX)==-1)*classes(2);
    outSampleE = 1 - mean(Y_test == Y_test_estimation);
else % multiclass, via one-vs-all method
    HX = zeros(size(X,1),nClasses);
    for J = 1:nClasses
        K_X_trainPs = feval(kernelFun,X,X_trainPs_cell{J,1}); 
        K_X_trainNg = feval(kernelFun,X,X_trainNg_cell{J,1}); 
        hX = mean(K_X_trainPs,2) - mean(K_X_trainNg,2) + B(J);
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
