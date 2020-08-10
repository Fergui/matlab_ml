function score = scoreAnalysis(theta,X,y,threshold)
%scoreAnalysis - Compute F1 score of theta using (X,y) data and threshold.
%   score = scoreAnalysis(theta,X,y,threshold) returns score which is an array
%   with accuracy, precision, recall and F1 scores.

pred = sigmoid(X*theta) >= threshold;
acc = mean(double(pred == y));
tp = sum((pred == 1) .* (y == 1));
pp = sum(pred == 1);
ap = sum(y == 1);
precis = (tp/pp);
recall = (tp/ap);
f1 = 2 * precis * recall / (precis+recall);
score = [acc,precis,recall,f1];

end