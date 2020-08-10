function plotResult(file)
%plotResult - Plot resulting fire arrival time from result file
%   plotResult(file) plots fire detections (X,y=1) and fire arrival time

r = load(file);
figure,
tignPlot(r.dtrain.X.*r.dtrain.sigma+r.dtrain.mu,r.dtrain.y,r.G,r.T)

end