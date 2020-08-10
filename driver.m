%% Driver for ML experiments

fprintf('\n>> Data Reading <<\n')
tic
data = csvread('polecreek.csv',1,0);
X = data(:,1:3);
y = data(:,4);
toc

fprintf('\n>> Data Pre-processing <<\n')
tic
[dtrain,dval,dtest] = preprocessData(X,y,[.8,.1,.1]);
toc

fprintf('\n>> Data Training <<\n')
tic
dtrain.lambda = .1; % regularization parameter
dtrain.scale = 1/3; % 1/(features * std(X)), where features=3 and std(X)=1
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
result.dtrain = dtrain;
result.dval = dval;
result.dtest = dtest;
save('result.mat','result')

fprintf('\n>> Plot Results <<\n')
figure, mesh(result.G{1},result.G{2},result.T);
hold on;
scatter3(X(y==1,1),X(y==1,2),X(y==1,3),'r*');
hold off;
