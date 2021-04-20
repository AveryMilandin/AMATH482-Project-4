# Digit-Classifiers

## Problem
In this project we are given images of hand drawn digits as well as the digits that they correspond to. The goal is to implement classifiers that can distinguish between different
digits using Linear Discriminant Analysis (LDA) and compare these classifiers to decision tree classifiers and support vector machine (SVM) classifiers that we will train using built-in MATLAB
functions. Since LDA does not work well for distinguishing between more than three digits at a time, we will create classifiers to distinguish between just two digits for every pair of
possible digits. For example, the first classifier we make will only be able to classify 0 and 1. We will also create one LDA classifier to distinguish between 3 different digits. We will
compare the acuuracy of these classifiers with decision tree and SVM classifiers that attempt to classify the same pairs of digits as our LDA classifiers as well as classifiers that
attempt to distinguish between all ten digits.

## Approach
Our training images are given in the file [train-images-idx3-ubyte.gz](train-images-idx3-ubyte.gz). We will read in these images as grayscale pixel matrices and then reshape them into vectors.
Each vector will make up one column of our data matrix. To create one LDA classifier to distinguish between 0 and 1, we will extract all columns of this matrix that correspond to a 
0 or a 1 and put them into their own matrix. We can then take the singular value decomposition of this matrix and extract the principal components of our image data, and then we can
project each of our images onto the first k principal components where we determine a reasonable value for k by examining the plot of singular values returned by the SVD. This gives us
two data clusters in k-dimensional space. With LDA we can find a line such that if we project both data clusters onto this line, the clusters will be as far apart from one another
and also have minimal inter class variance. From here we can determine a point on the line where everything on one side of the point gets classified as a 0 and everything on the other
side gets classified as a 1. We will repeat this training algorithm for each pair of possible digits and use a similar technique to classify between three digits.<br/>

After training classifiers with this method, we can feed in testing images from [t10k-images-idx3-ubyte.gz](t10k-images-idx3-ubyte.gz) to be classified and calculate the accuracy of
each classifier. We will then train decision tree and SVM classifiers on the same training data and test their accuracy in the same way.

## Results
With this approach the hardest pair of digits to classify with LDA classifiers was 3 and 5 with an accuracy of 87.6%. A decision tree classifier for the same digits had 95.8% accuracy,
and the SVM classifier had 96.7% accuracy. <br/>

The easiest digit pair to classify was 0 and 1, with the LDA classifier achieving 99.2% accuracy. The decision tree had 99.6%, and the SVM had 99.9%. <br/>

We trained a 3-digit LDA classifier to distinguis between 0, 1, and 8 that had an accuracy of 86.5%, whereas a decision tree to classify between all ten digits had 87.8% accuracy, and 
the SVM had 94.4% accuracy. <br/>

For a table of additional results for all classifiers and for a formal write up of the problem including a theoretical background, MATLAB algorithms, and a discussion of results see [Digit_Classification.pdf](Digit_Classification.pdf)
