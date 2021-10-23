#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:22:26 2020

@author: aadityabhatia
"""
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import unidecode
import argparse
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
import string
import sys
from nltk.corpus import stopwords

stops = stopwords.words("english")
punctuations = string.punctuation
for p in punctuations:
    stops.append(p)
stops.append('')

# arg parser
parser = argparse.ArgumentParser(description='Uses model files to test accuracy')
parser.add_argument('-l', '--lab_d', default='reference_data.csv')
parser.add_argument('-x', '--sample', default='f')
parser.add_argument('-d', '--dirName', default=None)

args = parser.parse_args()
dirName = args.dirName
    
if not dirName:
    current_directory = os.getcwd()
    dirName = f'{current_directory}/Data/'

dirHeirarchy = f'{dirName}Brands/'
Saves = f'{dirName}SavedModels/'
lab_d = args.lab_d


# Since we dont have new data, we get a sample of 20 data points from our labeled set
lab_df = pd.read_csv(lab_d)
df = lab_df.sample(20) if args.sample == 't' else lab_df


def getLemma(comment):
    lemmatized = []
    tokens = nlp(unidecode.unidecode(str(comment).lower()))
    for token in tokens:
        token = token.lemma_
        if token not in stops:
            lemmatized.append(token)
    return ' '.join(lemmatized)


def vectorize(brand_vocab_list, df):
    cv = CountVectorizer()
    word_count_vector = cv.fit(brand_vocab_list)
    sparse_matrix = word_count_vector.transform(df.query_lemma)
    df = pd.DataFrame(sparse_matrix.toarray(), columns=cv.get_feature_names())
    return df


def readModel(name):
    with open(dirName + name + '_model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(dirName + name + "_model.h5")
    # print("Loaded aggregate {} model from disk".format(name))
    return loaded_model


def getWordEmbeddings(df):
    cv = pickle.load(open(dirName + "countVectorizer.pkl", 'rb'))
    sparse_matrix = cv.transform(df.query_lemma)
    return pd.DataFrame(sparse_matrix.toarray(), columns=cv.get_feature_names())


def getPredictions(item, grouped, dl_model):
    # item = "brand" or "product"
    processed_df = pd.DataFrame()
    for name, dat in grouped:  # name = product or brand name
        # print(name, len(dat))
        dat = dat.reset_index()
        mini_model_name = '{}{}_model.sav'.format(Saves, name)

        if os.path.isfile(mini_model_name):
            with open('{}{}_vocab.txt'.format(Saves, name), 'rb') as fp:
                vocab_list = pickle.load(fp)
            X = vectorize(vocab_list, dat)
#            X['clicks'] = dat['clicks']
#            X['ctr'] = dat['ctr']
#            X['impressions'] = dat['impressions']

            model = pickle.load(open(mini_model_name, 'rb'))
            dat['{}_prediction'.format(item)] = model.predict(X)
            del model, X

        else:
            X = getWordEmbeddings(dat)
#            X['clicks'] = dat['clicks']
#            X['ctr'] = dat['ctr']
#            X['impressions'] = dat['impressions']

            X = X.to_numpy()
            model = dl_model
            model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            dat['{}_prediction'.format(item)] = model.predict_classes(X)
            del X, model
        processed_df = processed_df.append(dat)
        
    del dl_model
    return processed_df


# checking if labeled csv is correct
check_list = ['query', 'brand', 'product_name', 'product_id']
for item in check_list:
    if item not in df.columns:
        print(item, 'not found in the csv. Please submit correct')
        sys.exit()

df['query_lemma'] = df['query'].apply(getLemma)

# getting brands
grouped = df.groupby('brand')
brand_dl_model = readModel('brand')
brand_df = getPredictions('brand', grouped, brand_dl_model)
df = pd.merge(df, brand_df[['query', 'brand_prediction']], on='query')

# getting products
grouped = df.groupby('product_name')
product_dl_model = readModel('product')
product_df = getPredictions('product', grouped, product_dl_model )
df = pd.merge(df, product_df[['query', 'product_prediction']], on='query')

if 'brand_check' in df.columns and 'product_check' in df.columns:
    # getting the accuracy
    print("Brand accuracy for this sample is {}".format(len(df[df.brand_check == df.brand_prediction]) / len(df)))
    print("Product accuracy for this sample is {}".format(len(df[df.product_check == df.product_prediction]) / len(df)))

df.to_csv(dirName + 'Processed_' + lab_d)
