function K = gaussianKernel(X,Y,sigma)
%gaussianKernel - Generate Gaussian Kernel with width sigma
%   gaussianKernel(X,Y,sigma) returns rbf Kernel with width sigma 
%   between X and Y.
%   K(x,y) = exp(-||x-y||^2/sigma^2)

X2 = sum(X.^2, 2);
Y2 = sum(Y.^2, 2);
K = X2+Y2'-2*(X*Y');
K = exp(-1/(2*sigma^2)).^K;

end
