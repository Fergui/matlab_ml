%% Driver for ML experiments

in_file = 'polecreek.csv';
fprintf('\n>> Data Reading <<\n')
tic
data = csvread(in_file,1,0);
X = data(:,1:3);
y = data(:,4);
toc

fprintf('\n>> Data Pre-processing <<\n')
tic
[dtrain,dval,dtest] = preprocessData(X,y,[.9,.05,.05]);
toc

fprintf('\n>> Data Training <<\n')
tic
dtrain.lambda = 10; % regularization parameter
dtrain.scale = .1; % 1/(features * std(X)), where features=3 and std(X)=1
dtrain.theta = trainModel(dtrain);
toc

fprintf('\n>> Threshold Tunning <<\n')
tic
dtrain.threshold = tuneThreshold(dtrain,dval);
dtrain.split = log(dtrain.threshold/(1-dtrain.threshold));
toc

fprintf('\n>> Final Score <<\n')
tic
dtest.score = finalScore(dtrain,dtest);
toc

fprintf('\n>> Fire Arrival Time Estimation <<\n')
tic
[result.G,result.T] = tignEstimation(dtrain);
toc

fprintf('\n>> Save Results <<\n')
result.X = X;
result.y = y;
result.dtrain = dtrain;
result.dval = dval;
result.dtest = dtest;
out_file = sprintf('result_r%.2f_s%.2f.mat',dtrain.lambda,dtrain.scale);
save(out_file,'-struct','result')
fprintf('File %s saved\n',out_file)

%fprintf('\n>> Plot Results <<\n')
%figure, tignPlot(X,y,result.G,result.T);
