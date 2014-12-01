% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%

% A simple classification algorithm for the Kaggle MNIST competition


%----------------------- DATA FOR -----------------------------------------
%----------------------- TRAINING AND TESTING -----------------------------
%
% Assuming the data files train.csv and test.csv are in the current working
% directory:
trainSet = csvread('train.csv');
testSet = csvread('test.csv');
% File train.csv contains 42000 points with their labels (one of the 10 digits)
% Each point has 784 feature values corresponding to the intensity of each
% of the intensity of the corresponding pixel (784=28x28). The intensity is
% an integer value in [0,255].
% The labels (digit that each point corresponds to) is in the last column.
% File test.csv contains the same structure except that there are no labels
% (which have to be estimated). There are 28000 points in the test set.


%----------------------- k-FEATURES ---------------------------------------
%----------------------- FOR TRAINING DATA --------------------------------
%
% First step is to reduce the 784 features to only 10 features.
% The new features correspond to each of the 10 classes/digits in the data.
% So the i-th feature corresponds to i-th class, which is the digit i-1.
% The 10 features are determined by a parameter k, positive integer, so we
% call the features 'k-features'.
% Given a point x, given k, and given i (feature), this is how to compute 
% the value of the i-th k-features of x:
%       find the k closest points in the training set (not including x if x 
%       it is a training point) that belong to the i-th class, and take the
%       average of their distances to x (that's the value of the feature).

k = 5; % the kFeatures computed correspond to k=5. This value was selected 
% previously via cross-validation using leave-one-out on training set.

classes = unique(trainSet(:,end));
nClasses = length(classes);

nTrain = size(trainSet,1);
train_kFeat = zeros(nTrain, nClasses+1);
for I = 1:nTrain
    x = trainSet(I,1:end-1); % current point
    refPts = trainSet; % reference points
    refPts(I,:) = []; % because x is a training point, it is excluded from reference points
    for J = 1:nClasses
        d = classes(J); % digit
        idxs_d = refPts(:,end)==d; % points with that digit
        distances = sqrt(sum((refPts(idxs_d,1:end-1)-repmat(x,sum(idxs_d),1)).^2,2));
        sorted_distances = sort(distances);
        train_kFeat(I,J) = mean(sorted_distances(1:k));    
    end
end
train_kFeat(:,end) = trainSet(:,end);


%----------------------- k-FEATURES ---------------------------------------
%----------------------- FOR TESTING DATA ---------------------------------
%
nTest = size(testSet,1);
test_kFeat = zeros(nTest, nClasses);
refPts = trainSet; % reference points. Since these are testing points, all training points are used as reference for each x
for I = 1:nTest
    x = testSet(I,:); % current point
    for J = 1:nClasses
        d = classes(J); % digit
        idxs_d = refPts(:,end)==d; % points with that digit
        distances = sqrt(sum((refPts(idxs_d,1:end-1)-repmat(x,sum(idxs_d),1)).^2,2));
        sorted_distances = sort(distances);
        test_kFeat(I,J) = mean(sorted_distances(1:k));    
    end
end


%-------------------- ESTIMATIONS ON TEST SET ----------------------------
%--------------------------------------------------------------------------
% 
% With trainSet_kFeat and testSet_kFeat and training and testing points, 
% the next step is to classify the testing points with 15-Nearest Neighbors
% algorithm. The value 15 was pre-selected via cross-validation on the
% training set with leave-one-out method.
 
% Centering and normalizing trainSet
X_train(:,1:nClasses) = train_kFeat(:,1:nClasses,k);
means = mean(X_train,2); % centering row-wise
X_train = X_train - repmat(means,1,nClasses); % centering row-wise
norms = sqrt(sum(X_train.^2,2)); % normalizing row-wise 
norms(norms==0) = 1; % normalizing row-wise
X_train = X_train./repmat(norms,1,nClasses); % normalizing row-wise
Y_train(:,1) = train_kFeat(:,nClasses+1,k);

% Centering and normalizing testSet
X_test = test_kFeat;
means = mean(X_test,2); % centering row-wise
X_test = X_test - repmat(means,1,nClasses); % centering row-wise
norms = sqrt(sum(X_test.^2,2)); % normalizing row-wise 
norms(norms==0) = 1; % normalizing row-wise
X_test = X_test./repmat(norms,1,nClasses); % normalizing row-wise

Y_test_estimation = zeros(nTest,1);
q = 15;

for J = 1:nTest % estimations for each testing point
    disp(J);
	x = X_test(J,:); % current point
    ref_pts = X_train;
    ref_labels = Y_train;
    distancesTo_x = sqrt(sum((ref_pts-repmat(x,nTrain,1)).^2,2));
    sortedDistances = sort(distancesTo_x);
    ids = distancesTo_x <= sortedDistances(q);
    Y_test_estimation(J) = mode(ref_labels(ids));
end