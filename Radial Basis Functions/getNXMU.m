function NXMU = getNXMU(X,centers)
%this is the NxK matrix, with entry i,j equal to norm(x_i - mu_j)

N = size(X,1);
K = size(centers,1);

NXMU = zeros(N,K);% this is the NxK matrix, with entry i,j equal to norm(x_i - mu_j)
for J=1:K
    NXMU(:,J) = sqrt(sum((X - repmat(centers(J,:),N,1)).^2,2));
end

end

