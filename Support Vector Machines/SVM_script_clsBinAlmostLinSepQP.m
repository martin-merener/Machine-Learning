% Script implementing Support Vector Machines Explained, by Tristan Fletcher
% Case solved: classification - binary - Train Set is Almost Linearly Separable - Solution via QP.

d = 2; % dimension of space
N = 1000; % number of training points           

MaxIter_Data = 1000; % Matlab default is 200

% Creating target function: targetFun
a1 = 4*rand()-1;
a2 = 2*rand()-1;
a3 = (1/3)+(1/3)*rand();

% Generating points
X_train = rand(N,d); % training points
Y_train = 2*(a1*(X_train(:,1)-0.5).^2+a2*(X_train(:,1)-0.5)+a3 < X_train(:,2))-1; 

% Coefficients to solve the optimization problem:
C = 10;   % Note that C=Inf gives the algorithm for linearly separable.
H = (Y_train*Y_train').*(X_train*X_train');
f = -ones(N,1);
Aeq = Y_train';
beq = 0;
lb = zeros(N,1);
ub = repmat(C,N,1);

% Optimization: THOR or DELL
% options = optimoptions('quadprog');
% options = optimoptions(options,'Display', 'off');
% options = optimoptions(options,'Algorithm', 'interior-point-convex');
% options = optimoptions(options,'MaxIter', MaxIter_Data);
% [alpha,fval,exitflag,output,lambda] = quadprog(H,f,A,b,Aeq,beq,[],[],[],options);

% Optimization: TOSHIBA
options = optimset;
options = optimset(options,'Display', 'off');
options = optimset(options,'Algorithm', 'interior-point-convex');
options = optimset(options,'MaxIter', MaxIter_Data);
%[alpha,fval,exitflag,output,lambda] = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);
[alpha,fval,exitflag,output,lambda] = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);

% Coefficients for the hypothesis:
w = (alpha.*Y_train)'*X_train;
S = intersect(find(alpha>0.0001),find(alpha<=C)); % support vectors
n_S = length(S);
b = 0;
for J = 1:n_S
    b =  b + Y_train(S(J))-(alpha(S).*Y_train(S))'*(X_train(S,:)*X_train(S(J),:)');
end
b = b/n_S;

% Testing:
% Generating points
N_test = 5000;
X_test = rand(N_test,d); % training points
Y_test = 2*(a1*(X_test(:,1)-0.5).^2+a2*(X_test(:,1)-0.5)+a3 < X_test(:,2))-1; 
Y_guess = sign(X_test*w'+b);

accuracy = mean(Y_test==Y_guess);
disp(accuracy)                                  

figure                                                       
subplot(2,2,1);                                                           
gscatter(X_train(:,1),X_train(:,2),Y_train,'rb','oo',2);      
title('Training X -vs- Training Y');                                       

subplot(2,2,2);
gscatter(X_test(:,1),X_test(:,2),Y_test,'rb','oo',2);      
title('Testing X -vs- Testing Y');                    

subplot(2,2,3);                                              
gscatter(X_test(:,1),X_test(:,2),Y_guess,'rb','oo',2);        
title('Testing Set -vs- Estimated Y');                    

subplot(2,2,4);                                              
gscatter(X_test(:,1),X_test(:,2),Y_guess==Y_test,'kg','oo',2);         
title('Testing Set -vs- Correctness');  
