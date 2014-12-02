% Martin Merener, martin.merener@gmail.com, 02-Dec-2014 %
% ------------------------------------------------------%
function J = costOnly(XTheta, Y, R, N_features, lambda)
% Cost function associated to X and Theta.

[N_items,N_users] = size(Y);

X = reshape(XTheta(1:N_items*N_features), N_items, N_features);
Theta = reshape(XTheta(N_items*N_features+1:end), N_users, N_features);

values = X*Theta';
support = R==1;
J = 0.5*sum(sum((values(support)-Y(support)).^2)) + (lambda/2)*sum(sum(Theta.^2)) + (lambda/2)*sum(sum(X.^2));

end