% FUNCTION THAT CREATES A NEW child FROM parents.
function child = make_child(father,mother)

n = size(father,2);
crossover_point = randsample((1:n-1),1);
cointoss = randsample((1:2),1);
if cointoss == 1
    child = [father(1:crossover_point),mother(crossover_point+1:n)];
else
    child = [mother(1:crossover_point),father(crossover_point+1:n)];
end

