% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function W_vect = cell2vect(W,D)

lengths = (D(1:end-1)+1).*D(2:end);
ending = 0;
W_vect = zeros(sum(lengths),1);
for J = 1:length(lengths)
    start = ending+1;
    ending = start-1+lengths(J);
    W_vect(start:ending) = W{J,1}(:);
end

end