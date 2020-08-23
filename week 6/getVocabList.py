import re
from nltk.stem import PorterStemmer as ps

def func ():
#GETVOCABLIST reads the fixed vocabulary list in vocab.txt and returns a
#cell array of the words
#   vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt 
#   and returns a cell array of the words in vocabList.


## Read the fixed vocabulary list
    fid = open('vocab.txt')

# Store all dictionary words in cell array vocab{}
    n = 1899  # Total number of words in the dictionary

# For ease of implementation, we use a struct to map the strings => integers
# In practice, you'll want to use some form of hashmap
    vocabList = {}
    i=1
    for l in fid:
    # Actual Word
        vocabList[i]=l.split()
        i+=1
    fid.close()
    return vocabList;
