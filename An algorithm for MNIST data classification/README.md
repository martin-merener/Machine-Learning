
MNIST_script_simpleAlgo.m contains a simple yet effective approach to classify handwritten digits (MNIST data).

This algorithm classifies the testing points from the MNIST kaggle.com competition, with 97.89% accuracy. 

The algorithm first reduces the 784 features of a given target point into only 10 features (which takes ~ 0.5sec per point).
Next, the target is classified using nearest-neighbors based on these 10 features, and the corresponding 10 features of all the training points (previously computed), which takes ~ 0.05sec per point.

The target points (test set) are 28000, and the training points are 42000. 

More details in the script.


To contact me: martin.merener@gmail.com
