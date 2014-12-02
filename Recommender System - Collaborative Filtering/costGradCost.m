% Martin Merener, martin.merener@gmail.com, 02-Dec-2014 %
% ------------------------------------------------------%
function [J, grad] = costGradCost(XTheta, Y, R, N_features, lambda)
% Cost function and gradient associated to X and Theta.

[N_items,N_users] = size(Y);

X = reshape(XTheta(1:N_items*N_features), N_items, N_features);
Theta = reshape(XTheta(N_items*N_features+1:end), N_users, N_features);

values = X*Theta';
support = R==1;
J = 0.5*sum(sum((values(support)-Y(support)).^2)) + (lambda/2)*sum(sum(Theta.^2)) + (lambda/2)*sum(sum(X.^2));

ThetaTr = Theta';
X_grad = zeros(size(X));
for i = 1:size(X,1)
   idx = find(R(i,:)==1);
   ThetaTr_tmp = ThetaTr(:,idx);
   Y_tmp = Y(i,idx);
   X_grad(i,:) = (X(i,:)*ThetaTr_tmp-Y_tmp)*ThetaTr_tmp' + lambda*X(i,:);
end

Theta_grad_tr = zeros(size(Theta'));
for j = 1:size(ThetaTr,2)
   idx = find(R(:,j)==1);
   X_tmp = X(idx,:);
   Y_tmp = Y(idx,j);
   Theta_grad_tr(:,j) = X_tmp'*(X_tmp*ThetaTr(:,j)-Y_tmp) + lambda*ThetaTr(:,j);
end
Theta_grad = Theta_grad_tr';

grad = [X_grad(:); Theta_grad(:)];

end

% % Gradiant checking
% gradApprox = zeros(size(grad));
% epsilon = 0.0001;
% for J = 1:length(grad)
%     disp(J);
%     XThetaPlus = XTheta;
%     XThetaPlus(J) = XThetaPlus(J) + epsilon;
%     XThetaMinus = XTheta;
%     XThetaMinus(J) = XThetaMinus(J) - epsilon;
%     gradApprox(J) = (costGradCost(XThetaPlus, Y, N_features, lambda) - costGradCost(XThetaMinus, Y, N_features, lambda))/(2*epsilon);
% end
% scatter(grad,gradApprox);
