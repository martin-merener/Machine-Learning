% Martin Merener, martin.merener@gmail.com, 02-Dec-2014 %
% ------------------------------------------------------%


% Recommender system via Collaborative Filtering

clear all

% ---------------------- INPUTS -------------------------------------------
% -------------------------------------------------------------------------
% 
N_users = 50; % number of users
N_items = 100; % number of items that users may potentially rate (e.g., movies)
N_features = 10; % number of features about each item (e.g., for movies, the amount of action, comedy, drama, etc, in the movie
Y = max(-1,ceil(10*rand(N_items,N_users)-5)); % ratings: Y_ij is the rating 0 to 5 about item i-th by user j-th. *** NOTE: -1 represents no rating ***
R = min(0,Y)+1; % R_ij=1 indicates item i-th has been rated by user j-th
disp([N_features*(N_items + N_users),sum(R(:)==1)]); % number of coefficients to be determined, and number of known examples


% ---------------------- OUTPUTS ------------------------------------------
% ---------------------- TO LEARN -----------------------------------------
% 
% To learn:
% X of size N_items x N_features. For each item, the values of each of the N_fe feature
% Theta of size N_users x N_features. For each user, coefficients that multiplied to the features of an item (inner product) will estimate the user's rating of the item
% Ideally X would be known, but in practice it is assumed to be ignored, and learned.
% The goal is to learn X and Theta, and use them to recommend items to users, who are likely to rate the items highly.
% Once X and Theta are known, one way is to search for pairs user-item s.t. the user would rate the item highly, although user-item are not related yet.
% That is, once X and Theta are known, search for i (item) and j (user) maximizing X(i,:)*(Theta(j,:))', which is the estimated rating of user
% i-th on item j-th. The search should be restricted to pairs (i,j) such that user ith is not related to item j-th (e.g., a user that 
% did not watch a movie). For instance, if it is assumed that every user rates every item it is related to, then R gives the relation.
% Another way to recommend is to establish pairs of items that are similar: X(i_1,:) and X(i_2,:) are close, and pairs of users that are similar: 
% Theta(j_1,:) and Theta(j_2,:) are close, and then if user j_1 rates i_1 highly, then both i_1 and i_2 can be recommended to user j_2.

% The way to estimate X and Theta is via least squares. The ratings Y are known (N_items x N_users), and the error to minimize is 
% (Y(i,j) - X(i,:)*(Theta(j,:))')^2, for every (i,j) such that R(i,j)=1. So the cost function is the sum over (i,j), plus regularization term.
%
% Note that there are: N_features*(N_items + N_users) coefficients to determine, and sum(R(:)==1) known examples.

% Get optimals X and Theta
lambda = 0;
[Y_rowMeans0, rowMeansY] = getY_rowsMean0(Y,R);
[X,Theta,costs] = solveCollFilt_viaCG(Y_rowMeans0, R, N_features, lambda); % optimals X,Theta, and the sequence of costs until obtaining them.
Y_estimation_temp = X*Theta' + repmat(rowMeansY,1,N_users); % still real values
maxRating = max(max(Y(R==1))); % maximum possible rating
minRating = min(min(Y(R==1))); % minimum possible rating
Y_estimation = max(min(Y_estimation_temp,maxRating),minRating); % values made integers and within range of ratings. 
% Y_estimation = round(Y_estimation); % optional
avgRelErrorKnownRatings = mean(mean(abs(Y(R==1)-Y_estimation(R==1))/(maxRating-minRating))); % average relative error in the estimated ratings. Good to know.


% ---------------------- RECOMMENDATIONS ----------------------------------
% -------------------------------------------------------------------------
% 
% Having estimated X and Theta, now the algorithm can recommend to each user k items. These items are the k that have highest estimated rating
% for the given user, and that are not related to that user. k is a parameter.
RR = R==1; % With no loss of generality, it is assumed that every user rates every item it relates to. In practice, RR is a subset of R, since 
% users may relate to items but not rate them, although the system should not recommend items to which the user already related (and not rated). 
k = 15;
recommendations = cell(1,N_users); % for each user the list indicates the indices of the items recommended. A cell is used in case there are 
% less than k available items to recommend, which happens only if the user is related to less than k items.
for J = 1:N_users
    unrelatedItems = find(RR(:,J)==0); 
    N_unrelItems = length(unrelatedItems); % it may happen than N_unrelItems<k, in which case only N_unrelItems are recommended
    ratingsToConsider = -ones(N_items,1);
    ratingsToConsider(unrelatedItems) = Y_estimation(unrelatedItems,J); % estimated ratings for the user, with -1 for the related items.
    [~,items_sorted] = sort(ratingsToConsider); % items_sorted has the indices of the items sorted increasingly. We want the last k (if available)
    recommendations_J = items_sorted(end-min(k,N_unrelItems)+1:end); % the last k, or as many as available
    recommendations{1,J} = flipud(recommendations_J); % these are the indices of the k (or available) items recommended, starting by the highest ratings
end

