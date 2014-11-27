function Y = funTest(X)

Y = zeros(size(X));

a = -2;
b = 2;

ID_1 = X<a;
ID_2 = logical(X>=a) & logical(X<b);
ID_3 = X>=b;

Y(ID_1) = 2*sin(8*X(ID_1));
Y(ID_2) = X(ID_2).^3 - 4*X(ID_2); 
Y(ID_3) = abs(5*sin(4*X(ID_3)));

end

