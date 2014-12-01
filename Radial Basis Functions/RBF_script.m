% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%

% RADIAL BASIS FUNCTIONS (for classification and regression)
% 
% The hypothesis function is of the form: h(x) = sum_{j=1}^K w_j*R_j(norm(x-mu_j))
%
% where R_j is a non-negative-real valued function that depends on a parameter gamma_j. Example: R_j(t) = exp(-gamma_j*t^2)
% and mu_j are the K centers. The centers mu_j and then number K of them are determined using K-means. 
% The learning algorithm will produce the weights w_j's, and gamma_j.
%
% Estimations for regression are given by h(x).
% Estimations for binary classifications are given by sign(h(x)), assuming the labels are +1,-1. 
% Estimations for multi-class classification are given by one-vs-all approach (and binary classification).
%
% The approach to determine the weights is least squares, that is, minimizing sum_{i=1}^N ||y_i - h(x_i)||^2 (assuming gamma_j are known)
% This is the GLM approach, so if Phi is the NxK matrix: Phi_ij = R_j(norm(x_i-mu_j))
% then the optimal weights are given by w = pinv(Phi'*Phi)*Phi'*y
% 
% Finally, to determine the final gammas and weights we do the following:
% 
% Initialize parameters gamma_j
% Iterate:
%   1) For the given gammas, determine weights w
%   2) For the given weights, find optimal gammas
% 
% The optimality is with respect to sum_{i=1}^N ||y_i - h(x_i)||^2
% Step 2 is be solved via conjugate gradient.
% 
% The radial function can be arbitrary as long as the corresponding [PHI,GRAD] are computed in radialOnNorms.

clear all
problemToSolve = 'classification'; % 'classification' or 'regression'; % *** CURRENT generatingPoints4Regression works for d=1 only***

% -------------------- INITIALIZING ---------------------------------------
% -------------------- GENERATING DATA ------------------------------------
%
d = 2; % dimension of space
N = 1000; % number of training points           
N_test = 2000; % number of testing points
C = 3; % number of classes
if strcmp(problemToSolve,'classification')==1 % classification
    [X_data,Y_data,trueCenters] = generatingPoints4Classification(N+N_test,d,C); % generating points 
else
    [X_data,Y_data,trueCenters] = generatingPoints4Regression(N+N_test,d,C); % generating points 
end
idx_train = randsample(N+N_test,N); % subsetting points for training 
idx_test = setdiff(1:N+N_test,idx_train); % rest of the points for testing
X_train = X_data(idx_train,:); % training points
X_test = X_data(idx_test,:); % testing points
Y_train = Y_data(idx_train,:); % training labels
Y_test = Y_data(idx_test,:); % testing labels


% -------------------- DETERMINING CENTERS --------------------------------
% -------------------- USING K-MEANS --------------------------------------
%
X = X_train; % trainng points
K_max = 12; % maximum K to be considered
bestCosts = zeros(K_max,1); % best cost for each K (for each K we consider many centroid initializations)
bestIdx_cell = cell(K_max,1); % best clustering for each K
bestCentroids_cell = cell(K_max,1); % best centroids for each K

for K = 1:K_max 
    disp(K);
    [bestCost,bestIdx,bestCentroids] = kMeansClustering(X_train,K); % bestCentroids contains the centers for the RBF
    bestCosts(K) = bestCost;
    bestIdx_cell{K,1} = bestIdx;
    bestCentroids_cell{K,1} = bestCentroids;    
end
% % Look at the bestCosts to determine the optimal number of clusters K.
% scatter(1:K_max,bestCosts);
% % Or, look at the relative improvement (relative decrease in error) at K with respect to K-1.
% relativeImprovements = 1-bestCosts(2:K_max)./bestCosts(1:K_max-1);
% scatter(2:K,relativeImprovements);
K = 6; % CHOSEN number of centers
centers = bestCentroids_cell{K,1};
% % Centers vs Guessed Centers via K-Means
%  centersLabels = [zeros(size(trueCenters,1),1);ones(size(centers,1),1)];
%  gscatter([trueCenters(:,1);centers(:,1)],[trueCenters(:,2);centers(:,2)],centersLabels,'bgrcmyk','o',4);      
%  axis([0 1 0 1])
%  title('True centers vs Guesses centers via K-Means');                                       

tic;
% -------------------- RBF ALGORITHM --------------------------------------
% -------------------- LEARNING gammas and weights ------------------------
%
NXMU = getNXMU(X_train,centers);

if strcmp(problemToSolve,'regression')==1 % regression
    [gammas, w] = learnRBF(NXMU,Y_train); % learn parameters
else % classification
    classes = unique(Y_train); % find out classes
    nClasses = length(classes); % find out number of classes
    if nClasses == 2 % binary
        Y_temp = 2*(Y_train == classes(1))-1; % Transform labels to +1,-1
        [gammas, w] = learnRBFviaMyCG(NXMU,Y_temp); % learn parameters
    else % multiclass, via one-vs-all method
        GAMMAS = zeros(K,nClasses);
        W = zeros(K,nClasses);
        for J = 1:nClasses
            Y_temp = 2*(Y_train == classes(J))-1; % Transform labels to +1,-1
            [gammas, w] = learnRBFviaMyCG(NXMU,Y_temp); % learn parameters
            GAMMAS(:,J) = gammas;
            W(:,J) = w;
        end
    end
end
toc

% -------------------- ESTIMATION ON --------------------------------------
% -------------------- TRAINING SET ------------------------------------------
%
if strcmp(problemToSolve,'regression')==1 % regression
    [PHI,~] = radialOnNorms(NXMU,gammas);
    hX = PHI*w;
    Y_train_estimation = hX;
    inSampleE = mean(abs(Y_train_estimation-Y_train))/mean(abs(mean(Y_train)-Y_train));
else % classification
    if nClasses == 2 % binary
        [PHI,~] = radialOnNorms(NXMU,gammas);
        hX = PHI*w;
        Y_temp = sign(hX);
        Y_train_estimation = (Y_temp==1)*classes(1)+(Y_temp==-1)*classes(2);
        inSampleE = 1-mean(Y_train_estimation==Y_train);
    else % multiclass, via one-vs-all method
        HX = zeros(size(X_train,1),nClasses);
        for J = 1:nClasses
            [PHI,~] = radialOnNorms(NXMU,GAMMAS(:,J));
            hX = PHI*W(:,J);
            HX(:,J) = hX;
        end
        [~,idxClasses] = max(HX,[],2);
        Y_train_estimation = zeros(size(Y_train));
        for I = 1:length(Y_train_estimation)
            Y_train_estimation(I) = classes(idxClasses(I));
        end
        inSampleE = 1-mean(Y_train_estimation==Y_train);
    end
end


% -------------------- ESTIMATION ON --------------------------------------
% -------------------- TESTING SET ------------------------------------------
%
NXMU = getNXMU(X_test,centers);
if strcmp(problemToSolve,'regression')==1 % regression
    [PHI,~] = radialOnNorms(NXMU,gammas);
    hX = PHI*w;
    Y_test_estimation = hX;
    outSampleE = mean(abs(Y_test_estimation-Y_test))/mean(abs(mean(Y_test)-Y_test));
else % classification
    if nClasses == 2 % binary
        [PHI,~] = radialOnNorms(NXMU,gammas);
        hX = PHI*w;
        Y_temp = sign(hX);
        Y_test_estimation = (Y_temp==1)*classes(1)+(Y_temp==-1)*classes(2);
        outSampleE = 1-mean(Y_test_estimation==Y_test);
    else % multiclass, via one-vs-all method
        HX = zeros(size(X_test,1),nClasses);
        for J = 1:nClasses
            [PHI,~] = radialOnNorms(NXMU,GAMMAS(:,J));
            hX = PHI*W(:,J);
            HX(:,J) = hX;
        end
        [~,idxClasses] = max(HX,[],2);
        Y_test_estimation = zeros(size(Y_test));
        for I = 1:length(Y_test_estimation)
            Y_test_estimation(I) = classes(idxClasses(I));
        end
        outSampleE = 1-mean(Y_test_estimation==Y_test);
    end
end


% -------------------- DISPLAYING RESULTS ON ------------------------------
% -------------------- TRAINING AND TESTING -------------------------------
%
disp([inSampleE,outSampleE]);

if d==2 && strcmp(problemToSolve,'classification')==1
     
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


if d==1 && strcmp(problemToSolve,'regression')==1
    
    minX = min(min(X_train),min(X_test));
    maxX = max(max(X_train),max(X_test));
    minY = min([min(Y_train),min(Y_train_estimation),min(Y_test),min(Y_test_estimation)]);
    maxY = max([max(Y_train),max(Y_train_estimation),max(Y_test),max(Y_test_estimation)]);
    
    figure                                                       
    subplot(2,2,1);
    gscatter(X_train,Y_train); 
    axis([minX maxX minY maxY]);
    title('Training X vs Training Labels Y');                                       

    subplot(2,2,2);
    gscatter(X_train,Y_train_estimation);     
    axis([minX maxX minY maxY]);
    title(strcat('Training X vs Estimated Labels Y. In Sample Relative Error: ',num2str(100*inSampleE),'%'));                                            

    subplot(2,2,3);
    gscatter(X_test,Y_test);      
    axis([minX maxX minY maxY]);
    title('Testing X vs Testing Labels Y');                                       

    subplot(2,2,4);
    gscatter(X_test,Y_test_estimation);      
    axis([minX maxX minY maxY]);
    title(strcat('Testing X vs Estimated Labels Y. Out of Sample Relative Error: ',num2str(100*outSampleE),'%'));                                        
end