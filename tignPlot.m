function tignPlot(X,y,G,T)
%tignPlot - Plot resulting fire arrival time from trainning data (X,y)
%   tignPlot(X,y,G,T) plots fire detections (X,y=1) and fire arrival time

mesh(G{1},G{2},T);
hold on;
scatter3(X(y==1,1),X(y==1,2),X(y==1,3),'r.');
hold off;
xlabel('Longitude');
xlabel('Latitude');
xlabel('Time (days)');
legend('Fire estimation','Fire detections')

end