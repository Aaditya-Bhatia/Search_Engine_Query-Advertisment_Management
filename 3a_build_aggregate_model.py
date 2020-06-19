#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 00:07:41 2020
@author: aadityabhatia

Info: The module uses NLP techniques to clean text data and create word embeddings.
These word embeddings along with the data features are used as explanatory variables for a deep learning model
The deep learning model consists of 3 layers of neurons. The first two layers consist of 300 neurons each with the relu activation function. The final layer consists of a single neuron with the activation of
the sigmoid function, making binary predictions. 
Since this architecture is susceptible to overfitting, the procedure of dropouts has been used.
Dropout fraction is 0.5, i.e, 50% neurons are switched off during a training epoch.

"""

import pandas as pd
import string
import os
import unidecode
import nltk
from nltk.corpus import stopwords
import spacy
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from keras.utils import np_utils
from keras.callbacks import Callback

tqdm.pandas(desc="progress-bar")
import multiprocessing

cores = multiprocessing.cpu_count()
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing import sequence
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
import matplotlib.pyplot as plt
import pickle
import warnings
import argparse

nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
warnings.filterwarnings(action='ignore')
stops = stopwords.words("english")
punctuations = string.punctuation
for p in punctuations:
    stops.append(p)
stops.append('')


def getLemma(comment):
    lemmatized = []
    tokens = nlp(unidecode.unidecode(str(comment).lower()))
    for token in tokens:
        token = token.lemma_
        if token not in stops:
            lemmatized.append(token)
    return ' '.join(lemmatized)


def auc_roc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

    
def testTrainModel(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
    model = createModel()
    hist = model.fit(X_train, y_train, epochs=20, validation_split=0.3, batch_size=64, verbose=1)
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#    print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))
    # plotting the model fit wrt epochs
    fig, ax = plt.subplots()
    ax.plot(hist.history['accuracy'], label = 'Accuracy')
#    ax.plot(hist.history['auc_roc'], label = 'AUC')
    ax.set_xlabel('Epoch')
#    ax.legend(['AUC', 'Accuracy'], loc='upper right')
    return model


def tenFoldCrossValidation(X, Y):
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    testacc = []
    testauc = []
    acc = []
    auc = []

    for train, test in kfold.split(X, Y):
        # create model
        model = createModel()
        hist = model.fit(X[train], Y[train], epochs=15, validation_split=0.3, batch_size=64, verbose=0)

        # training
        acc.append(hist.history['accuracy'])
        auc.append(hist.history['auc_roc'])

        # testing set
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        # testing saving
        testacc.append(scores[1] * 100)
        testauc.append(scores[2] * 100)


def createModel():
    model = Sequential()
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    #    print(model.summary())
    return model


def makeVocab(df, df_vocab):
    # performing text normalization for various columns
    df['query_lemma'] = df['query'].apply(getLemma)
    df['brand_lemma'] = df['brand'].apply(getLemma)
    df['product_lemma'] = df['product_name'].apply(getLemma)
    df_vocab['db_brand_lemma'] = df_vocab['brand_name'].apply(getLemma)
    df_vocab['db_product_lemma'] = df_vocab['product_name'].apply(getLemma)

    vocabulary = []
    for i, row in df.iterrows():
        q_lst = row['query_lemma'].split()
        b_lst = row['brand_lemma'].split()
        p_lst = row['product_lemma'].split()
        vocabulary = vocabulary + q_lst + b_lst + p_lst
    for i, row in df_vocab.iterrows():
        db_lst = row['db_brand_lemma'].split()
        dp_lst = row['db_product_lemma'].split()
        vocabulary = vocabulary + dp_lst + db_lst

    vocabulary = list(set(vocabulary))
    return vocabulary


def saveModel(model, name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(dirName + name + "_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(dirName + name + "_model.h5")
    print("Saved {} model to disk".format(name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='''File to extract NLP features 
                                     from Database and Labelled data and make modules''')
    parser.add_argument('-v', '--df_vocab', default='master_vocabulary.csv')
    parser.add_argument('-l', '--df', default='reference_data.csv')
    parser.add_argument('-d', '--dirName', default=None)
    args = parser.parse_args()

    dirName = args.dirName
    if not dirName:
        current_directory = os.getcwd()
        dirName = f'{current_directory}/Data/'

    df = pd.read_csv(args.df)
    df_vocab = pd.read_csv(args.df_vocab)

	# making aggregate vocab
    vocabulary = makeVocab(df, df_vocab)

    # instantiate Count vectorizor
    cv = CountVectorizer()
    word_count_vector = cv.fit(vocabulary)
    pickle.dump(word_count_vector, open(dirName + "countVectorizer.pkl", "wb"))
    sparse_matrix = word_count_vector.transform(df.query_lemma)
    query_tf_df = pd.DataFrame(sparse_matrix.toarray(), columns=cv.get_feature_names())

    # removing sparse features
    X = query_tf_df
#    X['clicks'] = df['clicks']
#    X['ctr'] = df['ctr']
#    X['impressions'] = df['impressions']

    X = X.to_numpy()
    product_model = testTrainModel(X, Y=df.product_check.to_numpy())  # Making models here
    brand_model = testTrainModel(X, Y=df.brand_check.to_numpy())  # Making models here

    # to check the robustness of the model
    #    tenFoldCrossValidation(X, Y)

    saveModel(product_model, 'product')
    saveModel(brand_model, 'brand')
