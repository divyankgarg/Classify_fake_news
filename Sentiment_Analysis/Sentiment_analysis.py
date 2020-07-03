#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 00:41:55 2019

@author: shirishpandagare
"""

#Assignment 2


## Packages to be imported

import numpy as np 
import pandas as pd
import nltk 
from nltk import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

## defining a Class to perform the analysis

class SentimentAnalysis:
    
    def read_file(file_path):
        """Read the file and return in a format of numpy array"""
        corpus = pd.read_csv(file_path)
        corpus = np.array(corpus)
        return corpus


    def tokenization(corpus, method = "WhiteSpace"):
        """Perform row wise tokenization of the text using ["WhiteSpace", "TreeBank", "WordPunct"] methods.
        Desired method should be selected in the argument. Output is a list of tokenized word joined with the
        white-space along with a list of labels as a seperate variable"""
        
        data = corpus.T
        text = data[1]
        label = data[0]
        tokenizer_WhiteSpace = nltk.tokenize.WhitespaceTokenizer()
        tokenizer_TreeBank = nltk.tokenize.TreebankWordTokenizer()
        tokenizer_WordPunct = nltk.tokenize.WordPunctTokenizer()

        if method == "WhiteSpace":
            token = [tokenizer_WhiteSpace.tokenize(token) for token in text]
        elif method == "TreeBank":
            token = [tokenizer_TreeBank.tokenize(token) for token in text]
        elif method == "WordPunct":
            token = [tokenizer_WordPunct.tokenize(token) for token in text]
        else:
            print("""Please enter either ["WhiteSpace", "TreeBank", "WordPunct"] in method argument""")
        
        s = " "
        token = [[s.join(row)] for row in token]
            
        return token , label
    
    def token_normalization(token, method = "Lemmatize"):
        """Perform normalization by stemming and Lemmatization. 
        Try both ["Lemmatize", "Stemming"] method as they can out perform each other 
        in different cases."""
        
        stemmer_porter = nltk.stem.PorterStemmer()
        stemmer_lemmatizer = nltk.stem.WordNetLemmatizer()
        
        if method == "Lemmatize" :
            normalize_token = []
            for i in range(len(token)):
                l = [stemmer_lemmatizer.lemmatize(word) for word in token[i]]
                normalize_token.append(l)
        
        elif method == "Stemming":
            normalize_token = []
            for i in range(len(token)):
                l = [stemmer_porter.stem(word) for word in token[i]]
                normalize_token.append(l)      
        else:
            print("""Please enter either ["Lemmatize", "Stemming"] in method argument""")
            
        stop = stopwords.words('english') + list(string.punctuation)
        stopped_word_removed= []
        for i in range(len(normalize_token)):
            t = [word for word in normalize_token[i] if word not in stop]
            stopped_word_removed.append(t)
                
        s = " "
        token = [s.join(row) for row in stopped_word_removed]
        return token
    
    
    def feature_extraction(text ,n_gram_tuple, method = "BagOfWords"):
        """Perform feature extraction using Bag of Words or TI-IDF. Output is a sparse matrix converted in pandas
        DataFrame. number of n_gram required for the analysis shall be provided in the argument is a tuple format."""
        
        if method == "BagOfWords":
            vectorizer_BoW = CountVectorizer(ngram_range = n_gram_tuple)
            X = vectorizer_BoW.fit_transform(text)
        
        elif method == "TFIDF":
            vectorizer_Tfidf = TfidfVectorizer(ngram_range = n_gram_tuple)
            X = vectorizer_Tfidf.fit_transform(text)
            
        else:
            print("""Please enter either "BagOfWords" or "TFIDF" in method argument""")
            
        data = pd.DataFrame(X.toarray())
        return data
    
    
    def classifier(data, label, size):
        """Classification analysis is done using 4 different classifiers. Function returns the performance score
        of each of the classifier"""
        
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=size, random_state=42)
        
        # Bayesian Classifier 
        gnb = GaussianNB()
        y_pred = gnb.fit(X_train, y_train).predict(X_test)
        score1 = gnb.score(X_test, y_test)
        
        # Logisitic Regression Classifier
        logisitc_clf= LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
        clf = logisitc_clf.fit(X_train, y_train)
        score2 = clf.score(X_test, y_test)
        
        # Random Forest Classifier 
        Randomforest_clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
        rfc_clf = Randomforest_clf.fit(X_train, y_train)
        score3 = rfc_clf.score(X_test, y_test)
        
        # LDA 
        LDA = LinearDiscriminantAnalysis()
        LDA_clf = LDA.fit(X_train, y_train)
        score4 = LDA_clf.score(X_test, y_test)

        
        print("Naive Bayes Classifier score:- " + str(score1)), \
        print("Logistic Regression Classifier score:-" + str(score2)), \
        print("Random Forest Classifier score:-" + str(score3)), \
        print("LDA Classifier score:-" + str(score4))


## Applying the function of the dataset
        
# Loading text corpus 
file_path = "sentiment_analysis.csv"
corpus  = SentimentAnalysis.read_file(file_path)

# Tokenizing and normalizing the corpus
token, label = SentimentAnalysis.tokenization(corpus, "WhiteSpace")  #try using different methods and check the performance
Lemmatized_token_train = SentimentAnalysis.token_normalization(token, "Lemmatize")
stemmed_token_train = SentimentAnalysis.token_normalization(token, "Stemming")

# Feature extraction 
sparse_matrix_train = SentimentAnalysis.feature_extraction(Lemmatized_token_train, (1,2), method= "TFIDF") #try using different methods and check the performance

# Classifier 
SentimentAnalysis.classifier(sparse_matrix_train, label, 0.33)
