function F = featureCreation(X,X_supp,scale)
%featureCreation - Generate Kernel features with width scale for X with support X_supp
%   featureCreation(X,X_supp,scale) returns rbf Kernel features with  
%   width scale for X with supports X_supp including bias component and
%   negative linear time term:
%   h_theta(x,t) = theta_0 - theta_1*t + sum_j{K(x,X_supp_j)*theta_j}

m = size(X,1);
K = gaussianKernel(X(:,1:2),X_supp(:,1:2),scale);
F = [ones(m, 1) -X(:,3) K];

end