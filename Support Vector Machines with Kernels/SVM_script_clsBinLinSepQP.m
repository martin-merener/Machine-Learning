% Script implementing Support Vector Machines Explained, by Tristan Fletcher
% Case solved: classification - binary - Train Set Linearly Separable - Solution via QP.

d = 2; % dimension of space
N = 500; % number of training points           

MaxIter_Data = 200; % Matlab default is 200


% Creating target function: targetFun
Z = rand(d,d); % points that determine the hyperplane
P_in = Z(1,:);
G = Z(2:d,:)-repmat(P_in,d-1,1); % generators of the hyperplane's direction
P_out = 2*P_in; % point outside the hyperplane (?)
normal = P_out' - G'*pinv(G*G')*G*P_out'; % normal to hyperplane

% Generating points
X_train = rand(N,d); % training points
Y_train = sign((X_train-repmat(P_in,N,1))*normal); % given a point x outside the plane, the sign of (x-P_in)*normal determines wheter x lies on the plane or not

% Coefficients to solve the optimization problem:
H = (Y_train*Y_train').*(X_train*X_train');
f = -ones(N,1);
Aeq = Y_train';
beq = 0;
lb = zeros(N,1);
ub = [];

% Optimization: THOR or DELL (Ryerson)
% options = optimoptions('quadprog');
% options = optimoptions(options,'Display', 'off');
% options = optimoptions(options,'Algorithm', 'interior-point-convex');
% options = optimoptions(options,'MaxIter', MaxIter_Data);
% [alpha,fval,exitflag,output,lambda] = quadprog(H,f,A,b,Aeq,beq,[],[],[],options);

% Optimization: TOSHIBA
options = optimset;
options = optimset(options,'Display', 'off');
options = optimset(options,'Algorithm', 'interior-point-convex');
%[alpha,fval,exitflag,output,lambda] = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);
[alpha,fval,exitflag,output,lambda] = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);

% Coefficients for the hypothesis:
w = (alpha.*Y_train)'*X_train;
S = find(alpha>0.000001); % support vectors
n_S = length(S);
b = 0;
for J = 1:n_S
    b =  b + Y_train(S(J))-(alpha(S).*Y_train(S))'*(X_train(S,:)*X_train(S(J),:)');
end
b = b/n_S;

% Testing:
% Generating points
X_test = rand(N,d); % training points
Y_test = sign((X_test-repmat(P_in,N,1))*normal); % given a point x outside the plane, the sign of (x-P_in)*normal determines wheter x lies on the plane or not
Y_guess = sign(X_test*w'+b);

accuracy = mean(Y_test==Y_guess);
disp(accuracy)                                  

figure                                                       
subplot(1,2,1);                                                           
gscatter(X_train(:,1),X_train(:,2),Y_train,'br','xo');      
title('Training Set');                                       

subplot(1,2,2);                                              
gscatter(X_test(:,1),X_test(:,2),Y_guess,'br','xo');         
title('Testing Set with Guesses Labels');                    
