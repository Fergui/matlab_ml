function theta = trainModel(dtrain)
%trainModel - Train the model using train dataset
%   trainModel(dtrain) returns the trained parameters theta.

% Features creation
fprintf('> Creating kernel features...\n');
F = featureCreation(dtrain.X,dtrain.X_supp,dtrain.scale);
% Training the model
fprintf('> Training the model...\n');
theta = trainLogisticReg(F,dtrain.y,dtrain.lambda);

end