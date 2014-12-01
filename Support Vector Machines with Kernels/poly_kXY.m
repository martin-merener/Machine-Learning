% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function K = poly_kXY(X,Y)

a = 1;
b = 1;
c = 2;
K = (a*(X*Y') + b).^c;

end

