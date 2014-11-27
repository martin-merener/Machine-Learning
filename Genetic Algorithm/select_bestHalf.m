function index_individual = select_bestHalf(fitnesses)

    N = size(fitnesses,1);
    [~,I] = sortrows(1-fitnesses);
    index_individual = randsample(I(1:N/2),1);
end

