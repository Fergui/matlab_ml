function score = finalScore(dtrain,dtest)
%finalScore - Calculates final score using test data
%   finalScore(dtrain,dtest) calculates final score of using trained ML
%   model using test data.

fprintf('> Creating kernel features...\n');
F = featureCreation(dtest.X,dtrain.X_supp,dtrain.scale);
fprintf('> Calculating score...\n');
score = scoreAnalysis(dtrain.theta,F,dtest.y,dtrain.threshold);
score = score*100;
fprintf(' Final Accuracy = %f\n Precision = %f\n Recall = %f\n F1 = %f\n',score(1),score(2),score(3),score(4));

end