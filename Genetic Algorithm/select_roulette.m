% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function index_individual = select_roulette(fitnesses)

	total_fitness = sum(fitnesses);    
    roulette = total_fitness*rand();
    j=1;
    while sum(fitnesses(1:j))<roulette
    	j=j+1;
    end
    index_individual = j;
end

