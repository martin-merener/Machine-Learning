% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%

% In this script we compare SVM and PLA on linearly separable data

d = 2; % space dimension
N = 100; % number of training points

nRuns = 1000;
svmIsBetter = zeros(nRuns,1);

for JJ = 1:nRuns
    % create train set with labels, based on a random line separating the
    % points. The while loop makes sure there are points in both classes.
    setIsOK = 0;
    while setIsOK ==0
        P = 2*rand(2,1)-1; % point 1 determining the separating line
        Q = 2*rand(2,1)-1; % point 1 determining the separating line
        A = (P(2)-Q(2))/(P(1)-Q(1)); % slope of separating line
        B = Q(2) - a*Q(1); % constant of separating line
        X = 2*rand(N,2)-1; % training points
        Y = 2*(A*X(:,1)+B>X(:,2))-1; % labels
        if mean(Y)>-1 && mean(Y)<1 % assuring both classes +1,-1 are represented
            setIsOK = 1;
        end
    end

    % PLA learning
    maxIter = Inf;
    [w_pla, totIter] = perceptron_learningAlgo(X,Y,maxIter); % returns weights that determines the hypothesis, and the total number of iterations

    % SVM learning
    H = (Y*Y').*(X*X');
    f = -ones(N,1);
    Aeq = Y';
    beq = 0;
    lb = zeros(N,1);
    ub = [];
    % Optimization: TOSHIBA
    options = optimset;
    options = optimset(options,'Display', 'off');
    options = optimset(options,'Algorithm', 'interior-point-convex');
    [alpha,fval,exitflag,output,lambda] = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);
    % Coefficients for the hypothesis:
    w_svm = (alpha.*Y)'*X;
    S = find(alpha>0.000001); % support vectors
    n_S = length(S);
    b = 0;
    for J = 1:n_S
        b =  b + Y(S(J))-(alpha(S).*Y(S))'*(X(S,:)*X(S(J),:)');
    end
    b = b/n_S;

    % Testing
    N_test = 1000;
    X_test = 2*rand(N_test,2)-1; % training points
    Y_test = 2*(A*X_test(:,1)+B > X_test(:,2))-1; % labels

    % Testing PLA
    X_test_e = [ones(N_test,1), X_test];
    Y_est = zeros(size(Y_test));
    Y_est(X_test_e*w_pla>=0) = 1;
    Y_est(X_test_e*w_pla<0) = -1;
    disagreement_pla = 1-mean(Y_est == Y_test);

    % Testing SVM
    Y_est = sign(X_test*w_svm'+b);
    disagreement_svm = 1-mean(Y_est == Y_test);

    % Comparing
    svmIsBetter(JJ) = disagreement_svm<disagreement_pla;
end

mean(svmIsBetter)