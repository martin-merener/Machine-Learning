% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function direction = cgDirection(grad,gradBefore,directionBefore)
% calculates the direction by conjugate gradient algorithm

L = size(grad,1)+1; % number of layers
direction = reverse(grad); % first version, to be corrected below   

% vectorized version of gradBefore
gradBeforeVect = []; 
for J = 1:L-1
	gradBeforeVect = [gradBeforeVect; gradBefore{J,1}(:)];
end

% CG direction
if norm(gradBeforeVect)>0 
    % vectorized version of grad
    gradVect = [];
    for J = 1:L-1
        gradVect = [gradVect; grad{J,1}(:)];
    end
    momentum = gradVect'*(gradVect-gradBeforeVect)/(gradBeforeVect'*gradBeforeVect);
    % getting the actual conjugate gradient direction
    for J = 1:L-1
        direction{J,1} = direction{J,1} + momentum*directionBefore{J,1};
    end
end    

end

