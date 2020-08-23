import numpy as np
import processEmail
import svmTrain
from scipy.io import loadmat
from sklearn import svm
import getVocabList
import emailFeatures

## ==================== Part 1: Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.

print('\nPreprocessing sample email (emailSample1.txt)\n')

# Extract Features
data = open('emailSample1.txt')
file_contents = data.read()
word_indices  = processEmail.func(file_contents)

# Print Stats
print('Word Indices: \n')
print(word_indices)
print('\n\n')

## ==================== Part 2: Feature Extraction ====================
#  Now, you will convert each email into a vector of features in R^n. 
#  You should complete the code in emailFeatures.m to produce a feature
#  vector for a given email.

print('\nExtracting features from sample email (emailSample1.txt)\n')

# Extract Features
data = open('emailSample1.txt')
file_contents = data.read()
word_indices  = processEmail.func(file_contents)
features      = emailFeatures.func(word_indices)

# Print Stats
print('Length of feature vector:\n', features.shape[1])
print('Number of non-zero entries:\n', np.sum(features > 0))

## =========== Part 3: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.

# Load the Spam Email dataset
# You will have X, y in your environment
data1=loadmat('spamTrain.mat')
X=data1['X']
y=data1['y']

print('\nTraining Linear SVM (Spam Classification)\n')
print('(this may take 1 to 2 minutes) ...\n')

C = 1
model = svmTrain.func(X, y, C, 'linear')

p = model.predict(X)

j=0
for i in range(X.shape[0]):
    if p[i]==y[i,0]:
        j+=1
j=j/X.shape[0]

print('Training Accuracy:', j * 100)

## =================== Part 4: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat

# Load the test dataset
# You will have Xtest, ytest in your environment
data2=loadmat('spamTest.mat')
Xtest=data2['Xtest']
ytest=data2['ytest']

print('\nEvaluating the trained Linear SVM on a test set ...\n')

p = model.predict(Xtest)

j=0
for i in range(Xtest.shape[0]):
    if p[i]==ytest[i,0]:
        j+=1
j=j/Xtest.shape[0]

print('Test Accuracy:', j * 100)


## ================= Part 5: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.
#

# Sort the weights and obtin the vocabulary list
wei = model.coef_
id = np.argsort(wei)
wei = np.sort(wei)
weight= np.zeros(wei.shape)
idx=np.zeros(id.shape)
for i in range(wei.shape[1]):
    weight[0,i]=wei[0,wei.shape[1]-i-1]
    idx[0,i]=id[0,id.shape[1]-i-1]
    
vocabList = getVocabList.func()

print('\nTop predictors of spam: \n')
for i in range(15):
    print(vocabList[idx[0,i]][1], weight[0,i])

print('\n\n')

## =================== Part 6: Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included spamSample1.txt,
#  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
#  The following code reads in one of these emails and then uses your 
#  learned SVM classifier to determine whether the email is Spam or 
#  Not Spam

# Set the file to be read in (change this to spamSample2.txt,
# emailSample1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!
filename = 'emailSample2.txt'

# Read and predict
data3 = open(filename)
file_contents = data3.read()
word_indices  = processEmail.func(file_contents)
x             = emailFeatures.func(word_indices)
p = model.predict(x)

print('\nProcessed',filename,'\n\nSpam Classification:', p)
print('\n(1 indicates spam, 0 indicates not spam)\n\n')

