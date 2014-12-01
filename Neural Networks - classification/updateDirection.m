% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function W1 = updateDirection(W0,nu,direction)
L = size(W0,1)+1;
W1 = W0;
for J = 1:L-1 % updating weights
    W1{J,1} = W0{J,1} + nu*direction{J,1};
end
end
