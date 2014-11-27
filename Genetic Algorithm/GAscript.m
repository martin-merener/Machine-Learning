
clear all

n = 9;
N = 1000; % number of individuals in the population.
nGen = 50; % number of generations, including the initial one.
q = 0.05; % is a value in [0,1) indicating the % of best individuals that go directly to the following generation without any change at all. 
use_pre_existent_population = 'no';
if strcmp(use_pre_existent_population,'yes') == 1
    new_population = dlmread('last_population_decr.csv',',',0,0); % IF WE ALREADY HAVE A GOOD POPULATION.
else
    new_population = 10*rand(N,n); % A COMPLETELY NEW ONE. BUT THE RANGE OF THE VALUES DEPENDS ON THE APPLICATION.
end
[max_fitnesses, last_fitnesses_decr, last_population_decr] = genetic_algorithm(new_population,nGen,q,'roulette'); 

% INPUTS of genetic_algorithm:
    % new_population: is the matrix containing the individuals as rows.
    % nGen (defined above).
    % q (defined above).
    % a string denoting the method used to select the parents that will create the new generation of children: 'roulette', 'tournament', or 'bestHalf'.
% OUTPUTS of genetic_algorithm:
    % max_fitnesses: the maximum fitness of the population in each generation. If q*N>=1 then the values are non-decreasingly. 
    % last_fitnesses_decr: fitnesses of the last generation, ordered decreasingly. 
    % last_population_decr: population in the last generation ordered decreasingly according to their fitness.

dlmwrite('last_population_decr.csv',last_population_decr, 'delimiter', ',', 'precision', 9);