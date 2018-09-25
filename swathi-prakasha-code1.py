#!/usr/bin/python3
"""
Thomson Reuters Code challenge - Title Classification
@author: swathi Prakasha
The title classification is done by decision tree classifier using sklearn package.
"""
import re
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from nltk.corpus import stopwords
import nltk
from nltk.tokenize import RegexpTokenizer

token = RegexpTokenizer(r"\w+")
stopwords = nltk.corpus.stopwords.words("english")
wordVector = CountVectorizer(tokenizer=token.tokenize, stop_words=stopwords)
le = LabelEncoder()

#This function cleans given text
def cleanup(text):
    text = text.lower()
    text = re.sub('\s\W',' ',text)
    return text

#This functions predicts topic for the given title using model designed below.
def predictCat(title, model):
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    cod = model.predict(wordVector.transform([title]))
    return le.inverse_transform(cod)[0]

def main():
    #pass the input through arguments
    train1 =  sys.argv[1]
    test1   = sys.argv[2]

    #reads the train data and loads the cleaned data in to a dataframe
    data_set = pd.read_csv(train1)
    data_set= data_set.dropna(how='all')
    data_set['NEW_TITLE']  = [cleanup(sent) for sent  in data_set['TITLE']]

    #data preparation
    x_factor=  wordVector.fit_transform(data_set['NEW_TITLE'])
    y_factor = le.fit_transform(data_set['TOPIC'])

    #spliting up the trained data to create a model
    x_train, x_test, y_train, y_test = train_test_split(x_factor, y_factor, test_size=0.2)

    #Model building using Decision Tree Classifiers
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(x_train, y_train)

    #predicting the title for test data given
    test_data = pd.read_csv(test1, index_col = False)
    test_data['TEXT']  = [cleanup(s) for s in test_data['TITLE']]
    test_data['TOPIC']  = [predictCat(s, model) for s in test_data['TEXT']]
    test_data = test_data.drop(columns = ['TEXT'] )

    #loading the predicted data to csv
    test_data.to_csv('swathi-prakasha-result1.csv', index = False)
    print("Run sucessfully")

import sys
if len(sys.argv)<3:
    print("Enter a file name as input")
else:
    main()
