% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function cost = distortionCost(X,idx,centroids,K)
%returs the distortion cost

centroidsForEachX = zeros(size(X));
for J = 1:K
    idx_J = idx==J;
    m_J = sum(idx_J);
    centroidsForEachX(idx_J,:) = repmat(centroids(J,:),m_J,1);
end
cost = sum(sum((X - centroidsForEachX).^2,2),1);

end

