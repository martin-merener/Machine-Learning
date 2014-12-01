% Martin Merener, martin.merener@gmail.com, 01-Dec-2014 %
% ------------------------------------------------------%

function val = fitness(alpha) 

pre_val = sum((alpha - (1:9)).^2);
val = 1/(1+pre_val); % TO BE DEFINED DEPENDING ON THE APPLICATION.
