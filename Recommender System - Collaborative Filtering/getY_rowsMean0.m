% Martin Merener, martin.merener@gmail.com, 02-Dec-2014 %
% ------------------------------------------------------%
function [Y_rowMeans0, rowMeansY] = getY_rowsMean0(Y,R)
%Returns Y with mean 0 on each row, over their support, and the corresponding means.

N_items = size(Y,1);
Y_rowMeans0 = Y;
rowMeansY = zeros(N_items,1);

for J = 1:N_items
    support_J = R(J,:)==1;
    rowMeansY(J) = mean(Y(J,support_J));
    Y_rowMeans0(J,support_J) = Y(J,support_J) - rowMeansY(J);
end

end

