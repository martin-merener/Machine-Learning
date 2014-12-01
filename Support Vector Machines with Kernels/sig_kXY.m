% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function K = sig_kXY(X,Y)

a = 0.01;
b = 10;
K = tanh(a*(X*Y') + b);

end