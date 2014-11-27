function index_individual = select_tournament(fitnesses)

    N = size(fitnesses,1);
    candidates = randsample((1:N),2);
    if fitnesses(candidates(1),1) > fitnesses(candidates(2),1)
        index_individual = candidates(1);
    else 
        index_individual = candidates(2);
    end
end

