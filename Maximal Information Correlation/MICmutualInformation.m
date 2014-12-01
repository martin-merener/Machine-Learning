% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%

function MI = MICmutualInformation(P)
%This function computes the mutual information between two variables, given
%the join probability distribution P, assumed to be given as a matrix.
%       |p(x_1,y_1) ... p(x_1,y_m)|
%       |p(x_2,y_1) ... p(x_2,y_m)|
%  P =  |   ...     ...    ...    |
%       |p(x_n,y_1) ... p(x_n,y_m)|
%

Px = sum(P,2);
Py = sum(P,1);

PxPy = Px*Py;

A = P.*log(P./PxPy);

A(isnan(A)) = 0;

MI = sum(A(:));

end

