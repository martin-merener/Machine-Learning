% Script implementing Logistic Regression
% No regularization; using simple SGD.
% Source: Learning from Data, EdX course, Mostafa.

% Initialization
d = 2; % dimension of space
N_train = 2000; % number of training points           
N_test = 3000;
maxIter = 10000;
precision = 10^(-5); % stopping condition for change in norm(w0-w1)
nu = 1; % learning rate
threshold = 0.5; % Threshold used to estimate the class based on the estimated probabilities
sigmoid = @(x) 1./(1+exp(-x)); % Sigmoid function

% Creating target function, a binary function on the plane, with a polynomial boundary
a1 = 4*rand()-1;
a2 = 2*rand()-1;
a3 = (1/3)+(1/3)*rand();

% Generating points
X_train = rand(N_train,d); % training points
Y_train = 2*(a1*(X_train(:,1)-0.5).^2+a2*(X_train(:,1)-0.5)+a3 < X_train(:,2))-1; % training labels: +1 or -1

% Learning phase
X_e = [ones(N_train,1), X_train]; % adding bias term
w0 = zeros(d+1,1); % starting search of optimal weights for hypothesis
iter = 0;
wChange = precision+1; % measures current change in weights 
InErrors = zeros(maxIter,1); % in-sample errors per iteration
while iter < maxIter && wChange > precision  % iterates while wChange is large and iterations are few
    iter = iter+1;
    InErrors(iter) = -mean(log(sigmoid(Y_train.*(X_e*w0)))); % In-sample error ('cross-entropy error'). Slide 16/24, slides09 Mostafa.pdf
    J = randsample(N_train,1); % Takes a random point per iteration on which the gradient is computed
    grad = -(repmat(sigmoid(-Y_train(J).*(X_e(J,:)*w0)).*Y_train(J),1,d+1).*X_e(J,:))'; % As Slide 23/24, slides09 Mostafa.pdf, but only for 1 training pt.
    w1 = w0 - nu*grad; % update weights
    wChange = norm(w0-w1); % measure change in weights
    w0 = w1; % ready for next iteration   
end
InErrors = InErrors(1:iter); % Relevant errors in case iter < maxIter

% Generating test points
X_test = rand(N_test,d); % testing points
X_e = [ones(N_test,1), X_test]; % adding bias term
Y_test = 2*(a1*(X_test(:,1)-0.5).^2+a2*(X_test(:,1)-0.5)+a3 < X_test(:,2))-1; % testing labels: +1 or -1

% Estimation
Y_guess = 2*(sigmoid(X_e*w0)>threshold)-1; % sigmoid(X_e*w0)=P[y=1|x]. For the class estimation we use a threshold 

% Results
accuracy = mean(Y_test==Y_guess);
disp(accuracy);
OutError = -mean(log(sigmoid(Y_test.*(X_e*w0)))); % Out of sample error
disp(OutError);
disp(InErrors(end));

% PLOTS
figure                                                       
subplot(2,3,1);                                                           
gscatter(X_train(:,1),X_train(:,2),Y_train,'rb','oo',2);      
title('Training X -vs- Training Y');                                       

subplot(2,3,2);
gscatter(X_test(:,1),X_test(:,2),Y_test,'rb','oo',2);      
title('Testing X -vs- Testing Y');                    

subplot(2,3,3);                                              
gscatter(X_test(:,1),X_test(:,2),Y_guess,'rb','oo',2);        
title('Testing Set -vs- Estimated Y');                    

subplot(2,3,4);                                              
gscatter(X_test(:,1),X_test(:,2),Y_guess==Y_test,'kg','oo',2);         
title('Testing Set -vs- Correctness');  

subplot(2,3,5);                                              
plot(InErrors);
title('In-sample Errors');  