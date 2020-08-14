function theta = trainLogisticReg(X, y, lambda, option)
%trainLogisticReg - Trains logistic regression given a dataset (X, y) and a
%regularization parameter lambda
%   theta = trainLogisticReg(X, y, lambda) trains logistic regression using
%   the dataset (X, y) and regularization parameter lambda. Returns the
%   trained parameters theta.

if ~exist('options', 'var')
    option = 'fminunc';
end
% Initialize Theta
initial_theta = zeros(size(X, 2), 1); 
% Create "short hand" for the cost function to be minimized
costFunction = @(t) logisticRegCostFunction(t, X, y, lambda);
% Minimize using ...
switch option
    case 'fminunc'
        % ... fminunc built-in function
        options = optimset('Display', 'iter', 'MaxIter', 500, 'GradObj', 'on');
        theta = fminunc(costFunction, initial_theta, options);
    case 'fmincg'
        % ... fmincg from ML course
        options = optimset('MaxIter', 50, 'GradObj', 'on');
        theta = fmincg(costFunction, initial_theta, options);
    otherwise
        theta = initial_theta
        error('Bad minimization function specified.')
end

end
