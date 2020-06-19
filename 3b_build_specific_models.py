#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:44:09 2020

@author: aadityabhatia
"""

import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import argparse
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# arg parser
parser = argparse.ArgumentParser(description='Builds the models to test accuracy')
parser.add_argument('-d', '--dirName', default=None)

args = parser.parse_args()
dirName = args.dirName
if not dirName:
    current_directory = os.getcwd()
    dirName = f'{current_directory}/Data/'

dirHeirarchy = f'{dirName}Brands/'
Saves = f'{dirName}SavedModels/'


def testTrainAccuracyCheck(df, Y, name):
    X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.33)
    rf = RandomForestClassifier(n_estimators=250, n_jobs=-1, verbose=0)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, pred)
    if accuracy < 0.5:
        accuracy = 1 - accuracy
#     print("\nAccuracy of {} model: {:.4f}".format(name, accuracy))
    return accuracy


def getFeatureImportances(rf, X):
    print('\nGetting feature importances....\n')
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = X.columns
    print("Important words are:")
    for f in range(X.shape[1]):
        if f < 20:
            print(f + 1, names[indices[f]])
        else:
            break


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


if __name__ == "__main__":
    unprocessed_brands = []
    unprocessed_products = []
    for brand in os.listdir(dirHeirarchy):

        brand_path = os.path.join(dirHeirarchy, brand)
        if not os.path.isdir(brand_path):
            continue

        print("\nBuilding models for {}......".format(brand))

        # Vectorizing the query
        brand_df = pd.read_csv(os.path.join(brand_path, brand) + '.csv')
        with open('{}{}_vocab.txt'.format(Saves, brand), 'rb') as fp:
            brand_vocab_list = pickle.load(fp)
        df = vectorize(brand_vocab_list, brand_df)

        # training the random forest model

        Y = brand_df.brand_check
        X = df
#        X['clicks'] = brand_df['clicks']
#        X['ctr'] = brand_df['ctr']
#        X['impressions'] = brand_df['impressions']

        accuracy = testTrainAccuracyCheck(df, Y, brand)

        # checking if conditions are met
        if accuracy > 0.6 and Y.nunique() == 2:
#             print('saving {} model'.format(brand))
            model = getRandomForestModel(df, Y)
            pickle.dump(model, open('{}/{}_model.sav'.format(Saves, brand), 'wb'))
            # getFeatureImportances(model, df)
        else:
            print("Inadequate model features for brand:", brand)
            unprocessed_brands.append(brand)

        # making the models for product
        for product in os.listdir(brand_path):
            product_dir = os.path.join(brand_path, product)
            if os.path.isdir(product_dir):
                product_df = pd.read_csv(os.path.join(product_dir, product) + '.csv')
                # bilding the word vectors
                df = vectorize(brand_vocab_list, product_df)

                # preparing independent variables
                X = df
#                X['clicks'] = product_df['clicks']
#                X['ctr'] = product_df['ctr']
#                X['impressions'] = product_df['impressions']

                # preparing dependent variables
                Y = product_df.product_check

                # checking if we get a good fit
                accuracy = testTrainAccuracyCheck(df, Y, product)

                # checking if conditions are met
                if accuracy > 0.6 and Y.nunique() == 2:
                    print("\nBuilding models for {}......".format(product))
                    model = getRandomForestModel(df, Y)
                    pickle.dump(model, open('{}/{}_model.sav'.format(Saves, product), 'wb'))
                    # getFeatureImportances(model, df)
                else:
                    print("Inadequate model features for product:", product)
                    unprocessed_products.append(product)
