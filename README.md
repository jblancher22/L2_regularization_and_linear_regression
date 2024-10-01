# L2_regularization_and_linear_regression

This is a project I completed for my Data Mining class. I was given four sets of data, with each set containing a training file and a test file. 

For the first question (corresponding to file q2) I used the closed form solution for L2 regression to calculate the weights (w-vector). 
I then iterated lambda through a range of 0-150, finding both the training set Mean Squared Error (MSE) and the test set MSE. I then plotted a graph using matplotlib
with lambda as the x-axis and the MSE as the y-axis. For each data set, I found the lambda that minimized the test data's MSE and printed it at the bottom.

For the second question (corresponding to file q3), I did the same thing, except first cross-validated the data with k=10 folds.
