% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function [PHI,GRAD] = radialOnNorms(NXMU,gammas)
% The matrix NXMU of size NxK is defined as NXMU_ij = norm(x_i-mu_j)
% The matrix PHI of size NxK is defined as PHI_ij = R_j(norm(x_i-mu_j))
% The matrix GRAD of size NxK is defined as GRAD_ij = (delta/delta_j) R_j(norm(x_i-mu_j))

% gaussian. R_j(t) = exp(-gamma_j*t^2)
NXMUSq = NXMU.^2;
PHI = exp(-NXMUSq*diag(gammas)); % Note that: PHI_ij = R_j(norm(x_i-mu_j)) = exp(-gamma_j*norm(x_i-mu_j)^2)
GRAD = -NXMUSq.*exp(-NXMUSq*diag(gammas)); % Note that GRAD_ij = (delta/delta_j) R_j(norm(x_i-mu_j)) = -norm(x_i-mu_j)^2*exp(-gamma_j*norm(x_i-mu_j)^2)

% Other radial functions to consider, for which PHI and GRAD would have to
% be computed:
% 
% Exponential: R_j(t) = exp(-gamma_j*t)
% Rational: R_j(t) = 1 - (t^2)/(gamma_j + t^2)
% Multiquadric: R_j(t) = (t^2 + gamma_j^2)^(1/2)
% Inverse Multiquadric: R_j(t) = (t^2 + gamma_j^2)^(-1/2) 
% Cauchy: (1 + t^2/gamma_j)^(-1)
end

