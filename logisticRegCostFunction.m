function [J, grad] = logisticRegCostFunction(theta, X, y, lambda)
%logisticRegCostFunction - Compute cost and gradient for logistic regression with 
%regularization
%   logisticRegCostFunction(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
% Define hypothesis
h = sigmoid(X*theta);
% Define cost function
J = sum(-y.*log(h)-(1-y).*log(1-h))/m + lambda*theta(2:end)'*theta(2:end)/(2*m);
% Define gradient
grad = X'*(h-y)/m + lambda*[0;theta(2:end)]/m;
grad = grad(:);
% Define Hessian
%H = X'*diag(h)*X; 

end
