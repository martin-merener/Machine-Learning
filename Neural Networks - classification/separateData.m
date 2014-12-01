% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function [partOfData, reminderOfData] = separateData(data, p, useLabel)
% Returns all points labeled, partitioned into two subsets, the first one
% with p% of the points, and the second one with all the other points.
% It assumes that the last column contains labels.
% If useLabel == 1 then it will partition p% of each class, assuming that
% the last column contains the labels for the classes.
% If useLabel == 0 then it will partition regardless of the classes.

n = size(data,1);

if useLabel == 1
    classes = unique(data(:,end));
    nClasses = length(classes);
    partCell = cell(nClasses,1); 
    reminderCell = cell(nClasses,1); 
    for J = 1:nClasses
        C = classes(J);
        idxs_C = find(data(:,end) == C);
        n_C = length(idxs_C);
        idxs_some_C = randsample(idxs_C, ceil(p*n_C));
        idxs_rest_C = setdiff(idxs_C, idxs_some_C); 
        partCell{J,1} = data(idxs_some_C,:); 
        reminderCell{J,1} = data(idxs_rest_C,:); 
    end
    partOfData = cell2mat(partCell); 
    reminderOfData = cell2mat(reminderCell);
else
    idxs_some = randsample(1:n,ceil(p*n));
    idxs_rest = setdiff(1:n,idxs_some);
    partOfData = data(idxs_some,:);
    reminderOfData = data(idxs_rest,:);
end

end