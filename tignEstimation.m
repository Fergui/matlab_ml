function [G,T] = tignEstimation(dtrain)
%tignEstimation - Use trained model to estimate fire arrival time
%   [G,T] = tignEstimation(dtrain) returns the grid (G) and fire arrival
%   time (T).

fprintf('> Creating grid...\n');
dims = [500,500];
bbox = [min(dtrain.X(:,1)),max(dtrain.X(:,1)),min(dtrain.X(:,2)),max(dtrain.X(:,2))];
xp = linspace(bbox(1),bbox(2),dims(1));
yp = linspace(bbox(3),bbox(4),dims(2));
[Xp,Yp] = meshgrid(xp,yp);
X = [Xp(:), Yp(:)];
fprintf('> Creating kernel features...\n');
K = gaussianKernel(X(:,1:2),dtrain.X_supp(:,1:2),dtrain.scale);
fprintf('> Post-processing fire arrival time...\n');
T = (dtrain.theta(1)+K*dtrain.theta(3:end)-dtrain.split)/dtrain.theta(2);
T(T>max(dtrain.X(:,3))) = max(dtrain.X(:,3));
T = reshape(T,dims(1),dims(2))*dtrain.sigma(3)+dtrain.mu(3);
xx = Xp*dtrain.sigma(1)+dtrain.mu(1);
yy = Yp*dtrain.sigma(2)+dtrain.mu(2);
G = {xx, yy};

end