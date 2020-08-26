#  Practice exercises(with solutions) of Machine learning course by Andrew Ng translated into Python
![coursera.com/learn/machine-learning](https://www.mooclab.club/attachments/machine-learning-stanford-andrew-ng-course-png.1643/)


The repo contains all the practice exercises of Machine Learning Course on coursera.com taught by Andrew Ng. The course is taught in Octave/Matlab and the original practice problems are also written in Octave. The repository mostly provides line-by-line translation but preserves the overall logic used in the funtions. The names of the files have been kept the same for clarity.

The purpose of translating these codes is to create a code bank of some basic machine learning algorithms in python which is, debatably, the most popular language for data science.

# Exercise 1
Files in folder 'Week 1' implement linear regression. The file ex1.py implements linear regression in one variable on data in 'ex1data1.txt' while 'ex1_multi.py' implements linear regression in multiple variables on data in 'ex1data2.txt' with feature normalization and visualizations.

# Exercise 2
Files in folder 'Week 2' implement logistic regression. The file 'ex2.py' visualizes the data in 'ex2data1.txt' and classifies data using logistic regression while 'ex2_reg.py' visualizes data in 'ex2data2.txt' and implements logistic regression with regularization.

# Exercise 3
'ex3.py' in folder 'Week 3' implements multiclass logistic regression on data in 'ex3data1.mat'. While 'ex3_nn.py' implements the same multi class classification using a simple neural network with pre-determined weights in 'ex3weights.mat'.

# Exercise 4
'ex4.py' in folder 'Week 4' trains a neural network model on the same data as in the previous exercise titled here 'ex4data1.mat'. The data is set of images of numbers from 0 to 9 and the trained model learns to classify the image by different numbers.

# Exercise 5
'ex5.py' in folder 'Week 5' implements regularized linear regression to fit a curve to the data in 'ex5data1.mat'. It visualizes the data and the fit curve and does bias-variance analysis to judge the validity of the curve for the data. Next it extends linear regression to polynomial regression to include more features to fit a better curve to the data. It also checks validity of various learning parameters and also visualized all findings.

# Exercise 6
'ex6.py' in folder 'Week 6' classifies data in 'ex6data1.mat', 'ex6data2.mat' and 'ex6data3.mat' using support vector machines. Next 'ex6_spam.py' uses this technique to build a spam email classifier.

# Exercise 7
'ex7.py' in folder 'Week 7' implements K-means algorithm on unlabelled data and further uses this algorithm to compress an image. 'ex7_pca' implements principal component analysis(pca) for dimensionality reduction of data and then uses it on bigger dataset of 5000 face images.
