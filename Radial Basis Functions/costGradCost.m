% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function [cost,grad] = costGradCost(w,gammas,NXMU,Y)
% Computes the cost and its grad (with respect to gammas), everything else
% fixed, where the cost is defined as: mean(y_i-h(x_i)), mean taken over all
% training points, and h(x) = sum w_j*R_j(norm(x-mu_j)), with R_j depending
% on gamma_j

N = size(NXMU,1);
[PHI,GRAD] = radialOnNorms(NXMU,gammas);
hX = PHI*w;
cost = mean((hX-Y).^2);
grad = (2*(hX-Y)'*GRAD*diag(w)/N)';

end

% % Gradient checking
% epsilon = 0.0001;
% gradApprox = zeros(length(gammas),1);
% for J = 1:length(gammas)
%     gammasPlus = gammas;
%     gammasPlus(J) = gammasPlus(J)+epsilon;
%     gammasMinus = gammas;
%     gammasMinus(J) = gammasMinus(J)-epsilon;
%     gradApprox(J) = (costGradCost(w,gammasPlus,NXMU,Y) - costGradCost(w,gammasMinus,NXMU,Y))/(2*epsilon);
% end
% scatter(gradApprox,grad);

