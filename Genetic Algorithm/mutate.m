% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%

function child = mutate(child)

n=size(child,2);
prob_mutations = rand(1,n);
mutations = find(prob_mutations<(1/n));
n_mutations = size(mutations,2);
child(mutations) = 10*rand(1,n_mutations); % THIS DEPENDS ON THE APPLICATION.
