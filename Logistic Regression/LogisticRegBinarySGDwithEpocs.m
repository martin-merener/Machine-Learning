% Script implementing Logistic Regression
% No regularization; using simple SGD, and epocs
% Source: Learning from Data, EdX course, Mostafa.

% Initialization
d = 2; % dimension of space
N_train = 100; % number of training points           
N_test = 1000;
maxEpoc = 10000;
precision = 10^(-2); % stopping condition for change in norm(w0-w1)
nu = 1; % learning rate. Mostafa suggests nu=0.1
threshold = 0.5; % Threshold used to estimate the class based on the estimated probabilities
sigmoid = @(x) 1./(1+exp(-x)); % Sigmoid function

% Creating target function, a binary function on the plane, with a linear boundary
P1 = 2*rand(1,2)-1;
P2 = 2*rand(1,2)-1;

% Generating points
X_train = 2*rand(N_train,d)-1; % training points
Y_train = sign(diag((sqrt(sum((X_train - repmat(P1,N_train,1)).^2,2))).*repmat(norm(P2-P1),N_train,1))*((X_train - repmat(P1,N_train,1))*(P2-P1)')); % Using sign(cos). Training labels: +1 or -1

% Learning phase
X_e = [ones(N_train,1), X_train]; % adding bias term
w0 = zeros(d+1,1); % starting search of optimal weights for hypothesis
epoc = 0;
wChange = precision+1; % measures current change in weights
InErrors = zeros(maxEpoc,1); % in-sample errors per iteration
while epoc < maxEpoc && wChange >= precision  % iterates while wChange is large and iterations are few
    epoc = epoc+1;
    InErrors(epoc) = -mean(log(sigmoid(Y_train.*(X_e*w0)))); % In-sample error ('cross-entropy error'). Slide 16/24, slides09 Mostafa.pdf
    permutN = randsample(N_train,N_train); % a permulation to go through the training points, one at a time, computing gradient
    w0_epoc = w0; % starting search within epoc
    for J = permutN' % loop through the permutation for the current epoc
        grad = -(repmat(sigmoid(-Y_train(J).*(X_e(J,:)*w0_epoc)).*Y_train(J),1,d+1).*X_e(J,:))'; % As Slide 23/24, slides09 Mostafa.pdf, but only for 1 training pt.
        w1_epoc = w0_epoc - nu*grad; % update weights
        w0_epoc = w1_epoc; % ready for next iteration
    end
    w1 = w1_epoc; % just out of the epoc iteration 
    wChange = norm(w0-w1); % measure change in weights
    w0 = w1; % ready for next iteration     
end
InErrors = InErrors(1:epoc); % Relevant errors in case iter < maxIter

% Generating test points
X_test = 2*rand(N_test,d)-1; % testing points
X_e = [ones(N_test,1), X_test]; % adding bias term
Y_test = sign(diag((sqrt(sum((X_test - repmat(P1,N_test,1)).^2,2))).*repmat(norm(P2-P1),N_test,1))*((X_test - repmat(P1,N_test,1))*(P2-P1)')); % using sign(cos(angle)). Testing labels: +1 or -1

% Estimation
Y_guess = 2*(sigmoid(X_e*w0)>threshold)-1; % sigmoid(X_e*w0)=P[y=1|x]. For the class estimation we use a threshold 

% Results
accuracy = mean(Y_test==Y_guess);
disp(accuracy);
OutError = -mean(log(sigmoid(Y_test.*(X_e*w0))));
disp(InErrors(end));
disp(OutError);

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

