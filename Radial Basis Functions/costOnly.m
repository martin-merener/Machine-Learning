function cost = costOnly(w,gammas,NXMU,Y)
% Computes the cost, everything else fixed, where the cost is defined as: 
% mean(y_i-h(x_i)), mean taken over all training points, and 
% h(x) = sum w_j*R_j(norm(x-mu_j)), with R_j depending on gamma_j

N = size(NXMU,1);
[PHI,~] = radialOnNorms(NXMU,gammas);
hX = PHI*w;
cost = mean((hX-Y).^2);

end