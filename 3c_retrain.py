#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:03:07 2020

@author: aadityabhatia
"""

import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import argparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import unidecode
import spacy
import pickle
nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
import string
from nltk.corpus import stopwords

stops = stopwords.words("english")
punctuations = string.punctuation
for p in punctuations:
    stops.append(p)
stops.append('')

#add args here
#Saves = '/Users/aadityabhatia/Documents/GitHub.nosync/ad-management/Data/SavedModels/'
#df = pd.read_csv('/Users/aadityabhatia/Documents/GitHub.nosync/ad-management/Data/Labeled Data_20200410_v3.csv')
#df = df[df['brand'].str.contains('Acer|Samsung')] # get temp csv

# arg parser
parser = argparse.ArgumentParser(description='Uses model files to test accuracy')
parser.add_argument('-f', '--file', default='reference_data.csv')
parser.add_argument('-d', '--dirName', default=None)

args = parser.parse_args()
df = pd.read_csv(args.file)
dirName = args.dirName
if not dirName:
    current_directory = os.getcwd()
    dirName = f'{current_directory}/Data/'

dirHeirarchy = f'{dirName}Brands/'
Saves = f'{dirName}SavedModels/'

def testTrainAccuracyCheck(X, Y, name):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
    rf = RandomForestClassifier(n_estimators=250, n_jobs=-1, verbose=0)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, pred)
    if accuracy < 0.5:
        accuracy = 1 - accuracy
    return accuracy

def getRandomForestModel(df, Y):
    model = RandomForestClassifier(n_estimators=250, n_jobs=-1, verbose=0)
    model.fit(df, Y)
    return model

def vectorize(brand_vocab_list, brand_df):
    cv = CountVectorizer()
    word_count_vector = cv.fit(brand_vocab_list)
    sparse_matrix = word_count_vector.transform(brand_df.query_lemma)
    df = pd.DataFrame(sparse_matrix.toarray(), columns=cv.get_feature_names())
    return df

def getLemma(comment):
    lemmatized = []
    tokens = nlp(unidecode.unidecode(str(comment).lower()))
    for token in tokens:
        token = token.lemma_
        if token not in stops:
            lemmatized.append(token)
    return ' '.join(lemmatized)

def getVocab(comment):
    lemmatized = []
    tokens = nlp(unidecode.unidecode(str(comment).lower()))
    for token in tokens:
        token = token.lemma_
        if token not in stops:
            lemmatized.append(token)
    return lemmatized


if __name__ == "__main__":
    
    df['query_lemma'] = df['query'].apply(getLemma)
    df['query_vocab'] = df['query'].apply(getVocab)
    
    brands_grp = df.groupby('brand', as_index=0)
    
    for brand, brand_df in brands_grp:
        if len(brand_df) < 6:
            continue
        with open('{}{}_vocab.txt'.format(Saves, brand), 'rb') as fp:
            brand_vocab_list = pickle.load(fp)
        # updating the vocab
        brand_vocab_list = list(set(brand_vocab_list + brand_df['query_vocab'].sum()))
        #saving the updated vocab
        with open('{}{}_vocab.txt'.format(Saves, brand), 'wb') as fp:
            pickle.dump(brand_vocab_list, fp)

        # making embeddings
        X = vectorize(brand_vocab_list, brand_df)
#        X['clicks'] = brand_df['clicks']
#        X['ctr'] = brand_df['ctr']
#        X['impressions'] = brand_df['impressions']
        
        Y = brand_df.brand_check
        
        #getting the accuracy if acc < 0.6 we dont save the model
        accuracy = testTrainAccuracyCheck(X, Y, brand)
        # overriding the brand model 
        if accuracy > 0.6 and Y.nunique() == 2:
            model = getRandomForestModel(X, Y)
            pickle.dump(model, open('{}/{}_model.sav'.format(Saves, brand), 'wb'))
        
        # getting the products
        product_grp = brand_df.groupby('product_id')        
        for product, product_df in product_grp:
            if len(product_df) > 5:
                X = vectorize(brand_vocab_list, product_df)
        #        X['clicks'] = brand_df['clicks']
        #        X['ctr'] = brand_df['ctr']
        #        X['impressions'] = brand_df['impressions']
                Y = product_df.product_check
                accuracy = testTrainAccuracyCheck(X, Y, product)    
                # overriding the product model 
                if accuracy > 0.6 and Y.nunique() == 2:
                    print(accuracy, brand, product)
                    model = getRandomForestModel(X, Y)
                    pickle.dump(model, open('{}/{}_model.sav'.format(Saves, product), 'wb'))
       
