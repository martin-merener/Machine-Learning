function K = wave_kXY(X,Y) 
% Wave Wave is: k(x,y) = sin(norm(x-y)/a)/(norm(x-y)/a)
% The output here is: K_ij = k(X(i,:),Y(j,:))

a = 2;

n1 = size(X,1);
n2 = size(Y,1);
W = zeros(n1,n2);

for i=1:n2
    W(:,i) = (sum((repmat(Y(i,:),n1,1) - X).^2,2).^(1/2));
end

AUX = W./a;
K = sin(AUX)./AUX;

end
