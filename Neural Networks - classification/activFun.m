function [y,dy] = activFun(x)
% could be any smooth real-valued function with known analytical
% derivative, easy to code, fast to compute.

% tanh. use with labels pm1
%y = (exp(x) - exp(-x))./(exp(x) + exp(-x)); 
%dy = 1-y.^2;

% sigmoid. use with labels 0/1
y = 1./(1 + exp(-x)); 
dy = exp(-x).*(y.^2);

end
