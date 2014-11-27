% Script implementing 'Support Vector Machines Explained', by Tristan Fletcher
% Binary or Multiclass, non-linearly separable classes, via Kernels and QP.

kernelFun = 'gauss_kXY';
d = 2; % dimension of space
N = 500; % number of training points           
MaxIter = 1000; % Matlab default is 200
C = 100;   % Note that C=Inf gives the algorithm for linearly separable.

% Creating target function: targetFun
P1 = rand(1,2);
P2 = rand(1,2);
P3 = rand(1,2);
E1 = [1,0];
E2 = [-1,1]/sqrt(2);
E3 = [-1,-1]/sqrt(2);

% Generating points
X_train = rand(N,d); % training points
Y_train = zeros(N,1);
for J = 1:N
	% Neighbourhood of P1
    Q = X_train(J,:);
    Z = P1-Q;
    r = (((1+2*P1(1))*Z*E1'/norm(Z))^2+0.5)/2;
    if norm(Z)<r
        Y_train(J,1) = 1;
    else
        Y_train(J,1) = -1;
    end
 	% Neighbourhood of P2
    Q = X_train(J,:);
    Z = P2-Q;
    r = (((1+2*P2(1))*Z*E2'/norm(Z))^2+0.5)/4;
    if norm(Z)<r
        Y_train(J,1) = 1;
    else
       Y_train(J,1) = -1;
    end
  	% Neighbourhood of P3
    Q = X_train(J,:);
    Z = P3-Q;
    r = (((1+2*P3(1))*Z*E3'/norm(Z))^2+0.5)/8;
    if norm(Z)<r
        Y_train(J,1) = 1;
    else
       Y_train(J,1) = -1;
    end
end

% Coefficients to solve the optimization problem:
H = (Y_train*Y_train').*feval(kernelFun,X_train,X_train);
f = -ones(N,1);
Aeq = Y_train';
beq = 0;
lb = zeros(N,1);
ub = repmat(C,N,1);

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
options = optimset(options,'MaxIter', MaxIter);
%[alpha,fval,exitflag,output,lambda] = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);
[alpha,fval,exitflag,output,lambda] = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);

% Coefficients for the hypothesis:
S = intersect(find(alpha>0.0001),find(alpha<C)); % support vectors
b = mean(Y_train(S)) - mean(feval(kernelFun,X_train(S,:),X_train(S,:))*(Y_train(S).*alpha(S)));

% Testing:
% Generating points
N_test = 5000;
X_test = rand(N_test,d); % training points
Y_test = zeros(N_test,1);
for J = 1:N_test
	% Neighbourhood of P1
    Q = X_test(J,:);
    Z = P1-Q;
    r = (((1+2*P1(1))*Z*E1'/norm(Z))^2+0.5)/2;
    if norm(Z)<r
        Y_test(J,1) = 1;
    else
        Y_test(J,1) = -1;
    end
    % Neighbourhood of P2
    Q = X_test(J,:);
    Z = P2-Q;
    r = (((1+2*P2(1))*Z*E2'/norm(Z))^2+0.5)/4;
    if norm(Z)<r
        Y_test(J,1) = 1;
    else
       Y_test(J,1) = -1;
    end
    % Neighbourhood of P3
    Q = X_test(J,:);
    Z = P3-Q;
    r = (((1+2*P3(1))*Z*E3'/norm(Z))^2+0.5)/8;
    if norm(Z)<r
        Y_test(J,1) = 1;
    else
       Y_test(J,1) = -1;
    end
end

Y_guess = sign(feval(kernelFun,X_test,X_train)*(Y_train.*alpha)+b);

accuracy = mean(Y_test==Y_guess);
disp(accuracy);
disp(length(S));

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
