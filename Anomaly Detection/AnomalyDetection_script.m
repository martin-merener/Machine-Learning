% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%

% ANOMALY DETECTION
% Given a set of points, assumed to be normal (labeled) 0, and given a new
% point, the goal is to tell whether the new point is normal or if it is an
% anomaly.
% The approach here consists on estimating a density function for the
% training points (with label 0), and given a new point, estimate its
% probability (of being normal), and declare it an anomaly if the
% probability is below epsilon.
% The value of epsilon can be obtained through cross-validation, by
% maximizing the F1-score measure.

% --------------- SIMULATING NORMAL AND ABNORMAL POINTS -------------------
% ------------------------------------------------------------------------- 
d = 2; % dimension
% 
nNormal = 1000;
X1 = rand(nNormal,1); % 
X2 = X1+X1.*(1-X1).*normrnd(0,0.9,nNormal,1);
X_normal_d = [X1,X2]; % normal points to estimate the density
X1 = rand(nNormal,1); % 
X2 = X1+X1.*(1-X1).*normrnd(0,0.9,nNormal,1);
X_normal_v = [X1,X2]; % normal points to validate the epsilon threshold
nAbnormal = 30;
X3 = rand(nAbnormal,1);
X4 = (1-sign(X3-0.5))/2 + 0.2*rand(nAbnormal,1).*sign(X3-0.5) ;
X_abnormal = [X3,X4]; % abnormal points to estimate the density
X_alldata = [X_normal_d;X_normal_v;X_abnormal];
labels = [zeros(2*nNormal,1);ones(nAbnormal,1)];
% SEE PLOT: 
% gscatter(X_alldata(:,1),X_alldata(:,2),labels,'bgrcmyk','o',4);axis([0 1 0 1]);
      

% -------------------- DENSITY ESTIMATION ---------------------------------
% ------------------------------------------------------------------------- 
mus = mean(X_normal_d,1); % the means of each feature
Sigma = (1/nNormal)*(X_normal_d - repmat(mus,nNormal,1))'*(X_normal_d - repmat(mus,nNormal,1)); % covariance matrix
p = @(X) 1/((2*pi)^(d/2)*sqrt(det(Sigma)))*exp(-0.5*diag((X-repmat(mus,size(X,1),1))*pinv(Sigma)*(X-repmat(mus,size(X,1),1))'));


% -------------------- EPSILON-THRESHOLD SELECTION ------------------------
% ------------------------------------------------------------------------- 
p_normal = p(X_normal_v); % probabilities on the normal points for validation
p_abnormal = p(X_abnormal); % probabilities on the abnormal points
p_val = [p_normal;p_abnormal]; % all probabilities
y_val = [zeros(length(p_normal),1);ones(length(p_abnormal),1)]; % labels normal/abnormal for validation points

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

step = (max(p_val) - min(p_val))/1000;
for epsilon = min(p_val):step:max(p_val) % loop that finds the epsilon with highest F1 score
    
    y_estimation = p_val<epsilon; % estimation normal/abnormal according to the probability density and current epsilon
    prec = sum((y_estimation==1).*(y_val==1))/sum(y_estimation==1); % precision
    rec = sum((y_estimation==1).*(y_val==1))/sum(y_val==1); % recal
    F1 = (2*prec*rec)/(prec+rec); % F1-score
    
    if F1 > bestF1 % keeps the best F1-score and the best epsilon
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
    
end

y_estimation = p_val<bestEpsilon; % estimation normal/abnormal according to the probability density and BEST epsilon
false_normals = sum((y_estimation==0).*(y_val==1));
false_abnormals = sum((y_estimation==1).*(y_val==0));

if d==2
    
    figure                                                       
    subplot(2,2,1);
    scatter(X_normal_d(:,1),X_normal_d(:,2),3);      
    axis([0 1 0 1]);
    title('Normal points used to estimate the density');                                       

    subplot(2,2,2);
    gscatter([X_normal_v(:,1);X_abnormal(:,1)],[X_normal_v(:,2);X_abnormal(:,2)],y_val,'br','o',3);      
    axis([0 1 0 1]);
    title('Validation points vs True Labels (normal/abnormal)');                                       

    subplot(2,2,3);
    gscatter([X_normal_v(:,1);X_abnormal(:,1)],[X_normal_v(:,2);X_abnormal(:,2)],y_estimation,'br','o',3);      
    axis([0 1 0 1]);
    title(strcat('Validation points vs Estimated Labels (normal/abnormal). Best epsilon: ',num2str(bestEpsilon),'.', ' Best F1-score:',num2str(bestF1)));                                       

    subplot(2,2,4);
    gscatter([X_normal_v(:,1);X_abnormal(:,1)],[X_normal_v(:,2);X_abnormal(:,2)],y_val==y_estimation,'kg','o',3);      
    axis([0 1 0 1]);
    title(strcat('Validation points vs Misclassified points (in black).',' False normals:',num2str(false_normals),'. False abnormals:',num2str(false_abnormals)));                                       
    
    
end

