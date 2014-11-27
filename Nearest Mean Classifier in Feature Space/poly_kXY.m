function K = poly_kXY(X,Y)

a = 0.01;
b = 1;
c = 5;
K = (a*(X*Y') + b).^c;

end

