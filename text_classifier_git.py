#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:12:02 2019

@author: rupesh
"""

# -*- coding: utf-8 -*-
#Data can be downloaded at http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz


#Libraries
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files

#DataSet Import
reviews = load_files('txt_sentoken/')
X,y = reviews.data, reviews.target

#Pickle Store
with open('X.pickle','wb') as f:
    pickle.dump(X,f)
    
with open('y.pickle','wb') as f:
    pickle.dump(y,f)


#Corpus creation
corpus=[]
for i in range(len((X))):
    review = re.sub(r'\W',' ' ,str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+',' ',review)
    review = re.sub(r'^[a-z]\s+',' ',review)
    review = re.sub(r'\s+',' ',review)
    corpus.append(review)
    
#Create a Bag of words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features = 2000, min_df= 3, max_df=0.6 , stop_words= stopwords.words('english'))
X= vectorizer.fit_transform(corpus).toarray()
    
#Convert a BOW to TD-IDF
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

#Create a test train model
from sklearn.model_selection import train_test_split
text_train, text_test, sent_train,sent_test = train_test_split(X,y, test_size =0.2, random_state = 0)

#Fitting a Logistic regressiom
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)

#Predicting using the classifier
sent_pred = classifier.predict(text_test)

#Confusion Matrix to understand the result of the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test,sent_pred)

#Creating pickles to use files in further analysis

#Pickling the classifier
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)


#Pickling the vectorizer
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)









