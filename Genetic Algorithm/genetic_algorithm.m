% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function [max_fitnesses, last_fitnesses_decr, last_population_decr] = genetic_algorithm(new_population,nGen,q,select_criteria)
    % for input description see script.m

    N = size(new_population,1); % number of individuals in the population.
    n = size(new_population,2); % length of each individuals.
    max_fitnesses = zeros(nGen,1); % the maximum fitness of each generation.
    last_fitnesses_decr = zeros(N,1); % the fitnesses of the last generation ordered decreasingly. 

    generation = 1;
    while generation < nGen
        population = new_population; % current population.
        fitnesses = zeros(N,1); % will contain fitnesses of individuals in current population.
        for i = 1:N % now we compute the fitness of each individual.
            alpha = population(i,:); % current individual.
            fitnesses(i,1) = fitness(alpha); % fitness computation. THIS FUNCTION DEPENDS ON EACH APPLICATION.
        end
        max_fitnesses(generation,1) = max(fitnesses);
        [~,I] = sortrows(1-fitnesses); % indices decreasingly by fitness.
        % Now we make new_population with N individuals. 
        % The best q percent of the population will be in the new population.
        firsts_indiv = floor(q*N);
        new_population = zeros(N,n);
        new_population((1:firsts_indiv),:) = population(I(1:firsts_indiv),:);
        % Next we add (N - firsts_indiv) children, born from two parents from population. These two parents are chosen via 1 of the following criteria:
                % (1) roulette; (2) tournament; (3) bestHalf.
        for i = (firsts_indiv+1):N
            if strcmp(select_criteria,'roulette') == 1 
                index_father = select_roulette(fitnesses);  
                index_mother = select_roulette(fitnesses);
            end
            if strcmp(select_criteria,'tournament') == 1 
                index_father = select_tournament(fitnesses);
                index_mother = select_tournament(fitnesses);
            end
            if strcmp(select_criteria,'bestHalf') == 1 
                index_father = select_bestHalf(fitnesses);
                index_mother = select_bestHalf(fitnesses);
            end
            father = population(index_father,:);
            mother = population(index_mother,:);
            child = make_child(father,mother); % we use simple crossing. We could use more than 1 crossing point, and/or more than 2 parents.
            child = mutate(child); % each coordinate mutates with prob = 1/(length of individual) independently, with random value in a RANGE APP-DEPENDENT.
            new_population(i,:) = child;
        end
    	generation = generation+1;    
    end

	population = new_population; % current population.
    fitnesses = zeros(N,1); % fitnesses of individuals in current population.
    for i = 1:N % now we compute the fitness of each individual
        alpha = population(i,:); % current individual.
        fitnesses(i,1) = fitness(alpha);  
    end
    
    max_fitnesses(generation,1) = max(fitnesses);
    [~,I] = sortrows(1-fitnesses); % indices decreasingly by fitness.
    last_fitnesses_decr(:,1) = fitnesses(I,1);
    last_population_decr = population(I,:);
end