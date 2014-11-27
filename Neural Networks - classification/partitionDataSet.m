function [somePts, restPts] = partitionDataSet(dataSet, pct, useLabel)
% Returns all points labeled, partitioned into two subsets, the first one
% with pct% of the points, and the second one with all the other points.
% It assumes that the last column contains labels.
% If useLabel == 1 then it will partition pct% of each class, assuming that
% the last column contains the labels for the classes.
% If useLabel == 0 then it will partition regardless of the classes.

n = size(dataSet,1);

if useLabel == 0
    idxs_some = randsample(1:n,ceil(pct*n));
    idxs_rest = setdiff(1:n,idxs_some);
    somePts = dataSet(idxs_some,:);
    restPts = dataSet(idxs_rest,:);
else
    classes = unique(dataSet(:,end));
    nClasses = length(classes);
    someCell = cell(nClasses,1); 
    restCell = cell(nClasses,1); 
    for k = 1:nClasses
        C = classes(k);
        idxs_C = find(dataSet(:,end) == C);
        n_C = length(idxs_C);
        idxs_some_C = randsample(idxs_C, ceil(pct*n_C));
        idxs_rest_C = setdiff(idxs_C, idxs_some_C); 
        someCell{k,1} = dataSet(idxs_some_C,:); 
        restCell{k,1} = dataSet(idxs_rest_C,:); 
    end
    somePts = cell2mat(someCell); 
    restPts = cell2mat(restCell);
end

end