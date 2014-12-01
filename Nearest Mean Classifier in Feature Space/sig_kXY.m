% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function K = sig_kXY(X,Y)

a = 1;
b = 1;
K = tanh(a*(X*Y') + b);

end