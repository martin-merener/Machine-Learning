% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%
function P = distribution(X,Y,n_x,n_y)
%Given variables X,Y, P is the distribution on equally spaced n-by-m grid
%on the scatter plot of X vs Y.
%This only makes sense for X and Y having the same length 

N = length(X); % MUST be same as length(Y).
P = zeros(n_y,n_x);

x_I = min(X);
x_F = max(X);
d_x = (x_F-x_I)/n_x;
x_grid = (1:n_x)*d_x+x_I;

y_I = min(Y);
y_F = max(Y);
d_y = (y_F-y_I)/n_y;
y_grid = (1:n_y)*d_y+y_I;

for L = 1:N
    q_x = X(L); 
    q_y = Y(L); 
    ID_x = find(q_x<x_grid,1,'first');
    ID_y = find(q_y<y_grid,1,'first');
    P(n_y-ID_y+1,ID_x) = P(n_y-ID_y+1,ID_x)+1;
end

P = P/N;

end

