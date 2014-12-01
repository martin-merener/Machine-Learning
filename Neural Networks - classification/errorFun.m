% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function [costs, dcosts] = errorFun(y_guess,y_true)
% could be any smooth real-valued function with known analytical
% derivative, easy to code, fast to compute.

%costs = (y_guess - y_true).^2; % use either with pm1 or 0/1 labels
%dcosts = 2*(y_guess - y_true);

costs = sum(-y_true.*log(y_guess) - (1-y_true).*log(1-y_guess),2); % use only with labels 0/1
dcosts = (y_guess - y_true)./(y_guess.*(1-y_guess)); 

end

