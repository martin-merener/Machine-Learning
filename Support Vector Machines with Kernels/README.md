
Support Vector Machines with Kernels.


These files include 3 types of approaches to solve SVM (with Kernels):

i) Using Quadratic Programming (QP) on the dual. Rather limited in the number of training points, since it uses the Matlab solver for QP. E.g, "2000 training points"

ii) Using Newton to minimize the primal (using the gradient and the hessian). Can handle much more training points than QP but its limited by memory (requires inverting matrices that could be up to NxN size, N=|trainSet|). E.g, "15000 training points".

iii) Using Conjugate Gradient to minimize the primal. Can handle much more training points than Newton method. E.g., "50000 training points", perhaps even a lot more.


There are different kernels available for use.

To use these implementations you can start with any of the following scripts (increasing the level of complexity):

1) SVM_script_regLinQP.m: linear regression via QP  

2) SVM_script_regKernelQP.m: non-linear regression via QP

3) SVM_script_clsBinLinSepQP.m: binary classification for linearly separable data via QP

4) SVM_script_clsBinAlmostLinSepQP.m: binary classification for *almost* linearly separable data via QP

5) SVM_script_clsBinKernelQP.m: binary classification for non-linearly separable data via QP.

6) SVM_script_clsMultKernelQP.m: multiclass classification for non-linearly separable data via QP.

7) SVM_script_clsMultKernelNewton.m: binary/multiclass classification for non-linearly separable data via Newton minimization.

8) SVM_script_clsMultKernelCG.m: binary/multiclass classification for non-linearly separable data via Conjugate Gradient minimization.


Each script contains a source for the explanation behind the technique employed.

To contact me: martin.merener@gmail.com
