% This is a NN implementation following the approach from Mostafa course, Learning from Data course (EdX)
% Weight and Gradient are vectorized when possible.

% The learning algorithm has different options:
% 1) Binary or Multiclass (determined by the data).
% 2) If binary, labels could be [-1,1] or [0,1]. MAKE SAME CHOICE IN: (1) errorFun.m, (2) activFun.m accordingly.
% 3) The architecture: number of hidden layers and hidden units per layer can be chosen arbitrarily.
% 4) The mode in which the gradient is calculated: stochastic, or batch.
% 5) The learning rate calculation: steepest descent (only option).
% 6) The direction method: conjugate gradient (only option).
% 7) Type of regularization: 'decay' or 'elimination'; chosen inside costGradCostFun.m.
% 8) Other parameters: time bound; iteration bound; weight change bound; regularizer. 

% =========================================================================

clear all

% Read train set. Assume last column contains labels
trainSet = csvread('trainSet.csv'); 
%trainSet = centAndNormPlusFtrsLabel(trainSet, 1, 'n');

% binary labels 
binaryLabels = [0 1]; % either: [-1 1], or [0 1]. *** WITH MULTICLASS, USE 0,1 LABELS ***

% this particular train set is topological features on digits 0,...,9
% ad hoc relabeling for digits (0,...,9)
% trainSet(:,end) = (trainSet(:,end)>=5); % labels are now 0/1

% relabeling if required
% if isequal(binaryLabels,[-1 1])==1 % make labels into pm1 if required
%     trainSet(:,end) = 2*trainSet(:,end)-1;
% end

% Partition into train and test
pct = 0.20;
useLabel = 1;
[testSet, trainSet] = partitionDataSet(trainSet, pct, useLabel);

% Separate points from labels
X_train = trainSet(:,1:end-1);
Y_train = trainSet(:,end);
nTrain = length(Y_train);

% NN model architecture (number of hidden layers and hidden units per layer)
nFeatures = size(X_train,2);
D_hidden = [20 20];
classes = unique(Y_train);
nClasses = length(classes);
if nClasses == 2
    D = [nFeatures, D_hidden, 1]; 
else
    D = [nFeatures, D_hidden, nClasses];
end
L = length(D);
nW = (D(1:end-1)+1)*(D(2:end)'); % number of weights

% Making Y_train binary, if multiclass
if nClasses>2
    Y_train_e = zeros(size(Y_train,1),nClasses);
    for J = 1:nClasses
        Y_train_e(:,J) = Y_train == classes(J);
    end
else
    Y_train_e = Y_train;
end

% SPECIFICATIONS for learning algorithm
% type of gradient: (i) stochastic, (ii) batch
stochasticMode = 'no'; % 'yes' means stochasticMode, 'no' means batch mode
% CG implementation
CGimplementation = 'rasmussen'; % 'mostafa' or 'rasmussen'

% weights, and random initialization
W_init = cell(L-1,1);
maxNormSqX = max(sum(X_train.^2,2)); % largest squared norm over all points, which is 1 if X_train was normalized
for J = 1:L-1 
    epsilonInitial = 0.1/maxNormSqX; % could be 0.1 or any other value that is << 1.
    W_init{J,1} = normrnd(0,epsilonInitial,D(J)+1,D(J+1));
end
W_init = cell2vect(W_init,D);

% gradient descent
iterBound = 1000; % upper bound in numer of iterations
iter = 0; % initialize
timeBound = 360; % upper bound in the running time for the search (seconds)
deltaTime = 0; % initialize
WchangeBound = 10^(-6); % upper bound
Wchange = 1; % initialize
W0 = W_init; 
W1 = W_init;
lambda = 0.0001; % regularizer (lambda=0 is no regularization). costGradCostFun has 2 regularization options: (i) squared weight decay; (ii) weight elimination  
gradBefore = cell(L-1,1); % used for CG
directionBefore = cell(L-1,1); % used for CG
for J = 1:L-1 % used for CG
    gradBefore{J,1} = zeros(D(J)+1,D(J+1));
    directionBefore{J,1} = zeros(D(J)+1,D(J+1));
end
gradBefore = cell2vect(gradBefore,D);
directionBefore = cell2vect(directionBefore,D);
costsWithReg = zeros(iterBound,1);
bestCost = Inf;

tic;
warning('off','all');
while iter<iterBound && Wchange>=WchangeBound && deltaTime<timeBound
    iter = iter+1;
    deltaTime = toc;
    disp(iter);
    if strcmp(stochasticMode,'yes') % stochastic
        permutN = randsample(nTrain,nTrain); % a permulation to go through the training points, one at a time, computing gradient    
        W0_iter = W0;    
        for J = permutN  
            [cost,grad] = costGradCostFun_Vectorized(W0_iter,X_train(J,:),Y_train_e(J,:),lambda,nClasses,D);
            if cost<bestCost % keep best so far in your pocket
                bestCost = cost;
                W_final = W0_iter;
            end
            direction = -grad;
            if norm(gradBefore)>0 
                momentum = grad'*(grad-gradBefore)/(gradBefore'*gradBefore);
                direction = direction + momentum*directionBefore;
            end    
            gradBefore = grad;
            directionBefore = direction;
            [nu,W1_iter] = steepestLearning_Vectorized(W0_iter,X_train(J,:),Y_train_e(J,:),lambda,direction,D);
            W0_iter = W1_iter;
        end              
        W1 = W1_iter;  
    else % batch
        [cost,grad] = costGradCostFun_Vectorized(W0,X_train,Y_train_e,lambda,nClasses,D); % current cost (in-sample error) and its gradient   
        if cost<bestCost % keep best so far in your pocket
            bestCost = cost;
            W_final = W0;
        end
        direction = -grad;
        if norm(gradBefore)>0 
            momentum = grad'*(grad-gradBefore)/(gradBefore'*gradBefore);
            direction = direction + momentum*directionBefore;
        end    
        gradBefore = grad;
        directionBefore = direction;
        [nu,W1] = steepestLearning_Vectorized(W0,X_train,Y_train_e,lambda,direction,D);
    end
    Wchange = norm(W1-W0);
    W0 = W1; % ready for next iteration
    costsWithReg(iter) = cost;
end
warning('on','all');
% costs on train (with regularization)
costsWithReg = costsWithReg(1:iter);

W_final = vect2cell(W_final,D);

% forward propagation (train)
X = forwardProp(W_final,X_train);
hX = X{L,1}';

% cost on train (no regularization)
inSampleE = mean(sum(errorFun(hX,Y_train_e),2));

% accuracy on train (original classes)
Y_train_guess = zeros(size(Y_train));
for J = 1:nTrain
    [~,rowMax] = max(hX(J,:));
    Y_train_guess(J) = classes(rowMax);
end
accuracy_train = mean(Y_train == Y_train_guess);

% Separate points from labels (test)
X_test = testSet(:,1:end-1);
Y_test = testSet(:,end);

% forward propagation (test)
X = forwardProp(W_final,X_test);
hX = X{L,1}';

% Making Y_test binary, if multiclass
if nClasses>2
    Y_test_e = zeros(size(Y_test,1),nClasses);
    for J = 1:nClasses
        Y_test_e(:,J) = Y_test == classes(J);
    end
else
    Y_test_e = Y_test;
end

% cost test
outSampleE = mean(sum(errorFun(hX,Y_test_e),2));

% accuracy on test (original classes)
nTest = size(Y_test,1);
Y_test_guess = zeros(size(Y_test));
for J = 1:nTest
    [~,rowMax] = max(hX(J,:));
    Y_test_guess(J) = classes(rowMax);
end
accuracy_test = mean(Y_test == Y_test_guess);

disp([inSampleE, outSampleE]);
disp([accuracy_train, accuracy_test]);

% % % gradient checking
% grad_approx = cell(L-1,1);
% epsilon = 0.0001;
% Z = W_init;
% for J = 1:L-1
%     for K = 1:size(Z{J,1},1)
%         for LL = 1:size(Z{J,1},2)
%             disp([J,K,LL]);
%             Zplus = Z;
%             Zplus{J,1}(K,LL) = Zplus{J,1}(K,LL) + epsilon;
%             Zminus = Z;
%             Zminus{J,1}(K,LL) = Zminus{J,1}(K,LL) - epsilon;
%             grad_approx{J,1}(K,LL) = (costFunOnly(Zplus,X_train,Y_train,lambda) - costFunOnly(Zminus,X_train,Y_train,lambda))/(2*epsilon);
%         end
%     end
% end 
% [~, grad] = costGradCostFun(Z,X_train,Y_train,lambda,nClasses); % current cost (in-sample error) and its gradient   
% sumCheck = 0;
% for J = 1:L-1
%     sumCheck = sumCheck + sum(sum(abs(grad_approx{J,1}-grad{J,1})));
% end
% disp(sumCheck);

% we always work with hX in a range either: [0,1] for 0/1 labels, or [-1,1]
% for pm1 labels). Eventually we threshold hX either with: 0 for pm1 labels 
% or with 0.5 for 0/1 labels. 
