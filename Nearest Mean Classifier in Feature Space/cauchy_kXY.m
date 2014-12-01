% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function K = cauchy_kXY(X,Y) 
% Cauchy Kernel is: k(x,y) = 1/(1+norm(x-y)^2/a)
% The output here is: K_ij = k(X(i,:),Y(j,:))

a = 0.0001;

n1 = size(X,1);
n2 = size(Y,1);
W = zeros(n1,n2);

for i=1:n2
    W(:,i) = (sum((repmat(Y(i,:),n1,1) - X).^2,2).^(1/2));
end

AUX = (W.^2)/a;
K = 1./(1+AUX);

end
