function threshold = tuneThreshold(dtrain, dval, plotting)
%tuneThreshold - Tune theshold of decision using trained model and validation dataset.
%   tuneThreshold(dtrain,dval) returns the threshold that maximized F1 score.

if ~exist('plotting','var')
    plotting = false;
end
y = dval.y;
theta = dtrain.theta;
fprintf('> Creating kernel features...\n');
X = featureCreation(dval.X,dtrain.X_supp,dtrain.scale);
fprintf('> Tunning threshold using F1...\n');
levels = 4;
dt = .1; 
st = 0; 
et = 1;
for level=1:levels
    threshold_opts = st:dt:et;
    mt = numel(threshold_opts);
    score = zeros(mt,4);
    for t = 1:mt
        score(t,:) = scoreAnalysis(theta,X,y,threshold_opts(t));
    end
    [~,idx] = max(score(:,4));
    dt = dt/2;
    st = threshold_opts(max(1,idx-2));
    et = threshold_opts(min(idx+2,mt));
end
threshold = threshold_opts(idx);
disp_score = score*100;
fprintf(' Best threshold = %f\n Accuracy = %f\n Precision = %f\n Recall = %f\n F1 = %f\n',threshold,disp_score(idx,1),disp_score(idx,2),disp_score(idx,3),disp_score(idx,4));
% Plot results
if plotting
    figure, plot(threshold,disp_score(:,1),'r-');
    hold on;
    plot(threshold_opts,disp_score(:,2),'g-'); 
    plot(threshold_opts,disp_score(:,3),'b-'); 
    plot(threshold_opts,disp_score(:,4),'k-'); 
    hold off;
    legend('Accuracy','Precision','Recall','F1');
    title('Threshold Analysis');
    xlabel('threshold');
    ylabel('score');
end
end