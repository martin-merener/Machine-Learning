
Neural Networks for classification.

This was implemented following the approach in the EdX online course Learning From Data (introductory Machine Learning course) by Yaser S. Abu-Mostafa, Caltech professor.

To use this code start with NN_script.m

The code uses conjugate gradient, regularization, and can handle binary or multiclass data (not by reducing it to binary as in one-vs-all, but by handling all the classes at once).

Some specifications:

1) For the binary case labels could be {-1,1} or {0,1}. But MAKE SAME CHOICE IN: (i) errorFun.m, (ii) activFun.m accordingly.

2) The architecture of the netrwork (# hidden layers & # hidden units per layer) can be set arbitrarily in the variable "D_hidden".

3) The mode in which the gradient is calculated can chosen to be either: stochastic or batch.

4) The learning rate is calculated as steepest descent (largest decrease in the current direction).

5) The current direction is determined by Conjugate Gradient (Polak-Ribiere method).

6) Type of regularization: 'decay' or 'elimination'; chosen inside costGradCostFun.m.

7) Other parameters: time bound; iteration bound; weight change bound; regularizer. 


The directory includes a trainSet.csv file with data (42000) points, which first 10 columns are features and the last column is a class (digits: 0,...,9).


To contact me: martin.merener@gmail.com
