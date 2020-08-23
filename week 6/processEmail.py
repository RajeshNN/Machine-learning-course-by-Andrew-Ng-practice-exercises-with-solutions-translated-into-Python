import numpy as np
import re
from nltk.stem import PorterStemmer as ps
import getVocabList

def func(email_contents):
#PROCESSEMAIL preprocesses a the body of an email and
#returns a list of word_indices 
#   word_indices = PROCESSEMAIL(email_contents) preprocesses 
#   the body of an email and returns a list of indices of the 
#   words contained in the email. 
#

# Load Vocabulary
    vocabList = getVocabList.func()

# Init return value
    word_indices = []

# ========================== Preprocess Email ===========================

# Find the Headers ( \n\n and remove )
# Uncomment the following lines if you are working with raw emails with the
# full headers

# hdrstart = strfind(email_contents, ([char(10) char(10)]))
# email_contents = email_contents(hdrstart(1):end)

# Lower case
    email_contents = email_contents.lower()

# Strip all HTML
# Looks for any expression that starts with < and ends with > and replace
# and does not have any < or > in the tag it with a space
    email_contents = re.sub('<[^<>]+>', "" ,email_contents)

# Handle Numbers
# Look for one or more characters between 0-9
    email_contents = re.sub('[0-9]+', 'number', email_contents)

# Handle URLS
# Look for strings starting with http:// or https://
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)

# Handle Email Addresses
# Look for strings with @ in the middle
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)

# Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)


# ========================== Tokenize Email ===========================

# Output the email to screen as well
    print('\n==== Processed Email ====\n\n')


    # Tokenize and also get rid of any punctuation
    str = re.split("[\W ]", email_contents)

    l=0
    
    # Stem the word 
    for w in str:

        w=re.sub('[^a-zA-Z0-9]', "", w)

        w=ps().stem(w)
        
    # Skip the word if it is too short
        if len(w) < 1:
            continue
    
    # Look up the word in the dictionary and add to word_indices if
    # found
        for i in range(1,len(vocabList)+1):
            if w==vocabList[i][1]:
                word_indices.append(i)
        if (l + len(w) + 1) > 70:
            print('\n')
            l = 0;
    
        print(w,end=' ')
        l = l + len(w) + 1;

# Print footer
    print('\n\n=========================\n')

    return word_indices;
