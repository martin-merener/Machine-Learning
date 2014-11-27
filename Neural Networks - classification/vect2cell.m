function W = vect2cell(W_vect,D)

L = length(D);
W = cell(L-1,1);
starting = 0;
for J = 1:L-1
    values = W_vect(starting+1:starting+(D(J)+1)*D(J+1));
    W{J,1} = reshape(values,D(J)+1,D(J+1));
    starting = starting+(D(J)+1)*D(J+1);
end

end

