function [theta,threshold,score] = trainModel(dtrain,dval)
%trainModel - Train the model using train dataset
%   trainModel(dtrain,dval) returns the trained parameters theta.

ns = length(dtrain.scale_opts);
nl = length(dtrain.lambda_opts);
theta = zeros(ns,nl);
threshold = zeros(ns,nl);
score = zeros(ns,nl);
for is=1:ns
    scale = dtrain.scale_opts(is);
    % Features creation
    fprintf('> Creating kernel features with scale=%f...\n',scale);
    F_train = featureCreation(dtrain.X,dtrain.X_supp,scale);
    F_val = featureCreation(dval.X,dtrain.X_supp,scale);
    for il=1:nl
        lambda = dtrain.lambda_opts(il);
        % Training the model
        fprintf('> Training the model with regularization parameter %f...\n',lambda);
        theta(is,il) = trainLogisticReg(F_train,dtrain.y,lambda);
        threshold(is,il),score(is,il) = tuneThreshold(theta(is,il),F_val,dval.y);
    end
[score,idx] = max(score_opts);
theta = theta(idx);
threshold = threshold(idx);

end