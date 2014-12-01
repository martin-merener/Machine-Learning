% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function direction = reverse(grad)
L = size(grad,1)+1;
direction = grad;
for J = 1:L-1 % updating weights
    direction{J,1} = -grad{J,1};
end
end