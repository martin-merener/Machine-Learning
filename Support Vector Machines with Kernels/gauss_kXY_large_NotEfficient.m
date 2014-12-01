% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function K = gauss_kXY_large(X,Y) 
% Computes the Kernel matrix when the number of points is large, but subdividing 
% the input matrices into submatrices and putting the resulting pieces together.
% Gaussian Kernel is: k(x,y) = exp(-norm(x-y)^2/(2a^2))
% The output here is: K_ij = k(X(i,:),Y(j,:))

chunkSize = 20000; % each submatrix is not larger (rows) than this
nX = size(X,1); % number of points
nY = size(Y,1); % number of points

if nX<=chunkSize && nY<=chunkSize
    K = gauss_kXY(X,Y); % just usual    
else
    subXsize = ceil(nX/ceil(nX/chunkSize)); % subdivision size for X
    subYsize = ceil(nY/ceil(nY/chunkSize)); % subdivision size for Y
    cutsX = 0:subXsize:nX; % points for subdivision for X
    cutsX(end) = nX; % in case it falls short
    cutsY = 0:subYsize:nY; % points for subdivision for Y
    cutsY(end) = nY; % in case it falls short
    nSubX = length(cutsX)-1; % number of subdivisions for X
    nSubY = length(cutsY)-1; % number of subdivisions for Y    
    K = zeros(nX,nY);
    for I = 1:nSubX
        for J = 1:nSubY
            K(cutsX(I)+1:cutsX(I+1),cutsY(J)+1:cutsY(J+1)) = gauss_kXY(X(cutsX(I)+1:cutsX(I+1),:),Y(cutsY(J)+1:cutsY(J+1),:));
        end
    end
end
    
end