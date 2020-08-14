function [dtrain, dval, dtest] = preprocessData(X,y,split)
%preprocess - Pre-process data to run ML on satellite detections
%   preprocess(X,y,split) returns train, validation, and test data cleaned,
%   normalized, and including supports for Kernel generation.

fprintf('> Cleaning the data...\n');
X_fire = X(y==1,:);
X_nofire = X(y==0,:);
b = .01;
mask = false(size(X_nofire,1),1);
X_b_min = X_fire(:,1:2)-b;
X_b_max = X_fire(:,1:2)+b;
for i = 1:size(X_fire,1)
    mask_i = logical((X_nofire(:,1) >= X_b_min(i,1)).*(X_nofire(:,1) <= X_b_max(i,1)).*(X_nofire(:,2) >= X_b_min(i,2)).*(X_nofire(:,2) <= X_b_max(i,2)).*(X_nofire(:,3) > X_fire(i,3)));
    mask = mask | mask_i;
end
X = [X_fire; X_nofire(~mask,:)];
y = [ones(size(X_fire,1),1); zeros(size(X_nofire(~mask,:),1),1)];

fprintf('> Splitting the data...\n');
m = size(X,1);
idxs = randperm(m);
s1 = cast(m*split(1),'int32');
s2 = cast(m*sum(split(1:2)),'int32');
dtrain.X = X(idxs(1:s1),:);
dtrain.y = y(idxs(1:s1));
dtrain.m = size(dtrain.X,1);
dval.X = X(idxs(1+s1:s2),:);
dval.y = y(idxs(1+s1:s2));
dval.m = size(dval.X,1);
dtest.X = X(idxs(1+s2:m),:);
dtest.y = y(idxs(1+s2:m));
dtest.m = size(dtest.X,1);

fprintf('> Normalizing the features...\n');
[dtrain.X, mu, sigma] = featureNormalize(dtrain.X); 
dval.X = (dval.X - mu)./sigma;
dtest.X = (dtest.X - mu)./sigma;
dtrain.mu = mu;
dtrain.sigma = sigma;

fprintf('> Creating supports...\n');
nsupp = sum(dtrain.y==1);
idxs = randperm(dtrain.m-nsupp,nsupp);
X_train_1 = dtrain.X(dtrain.y==1,:);
y_train_1 = dtrain.y(dtrain.y==1);
X_train_0 = dtrain.X(dtrain.y==0,:);
y_train_0 = dtrain.y(dtrain.y==0);
dtrain.X_supp = [X_train_0(idxs,:);X_train_1];
dtrain.y_supp = [y_train_0(idxs);y_train_1];
end
