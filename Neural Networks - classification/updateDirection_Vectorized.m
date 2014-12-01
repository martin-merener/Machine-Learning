% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function W1 = updateDirection_Vectorized(W0,nu,direction)
W1 = W0 + nu*direction;
end
