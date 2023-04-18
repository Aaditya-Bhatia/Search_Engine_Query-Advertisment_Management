#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 12:54:07 2020

@author: aadityabhatia
"""

import pandas as pd
import os
import spacy
import unidecode
from nltk.corpus import stopwords
import string
import pickle
import argparse

nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
stops = stopwords.words("english")
punctuations = string.punctuation
for p in punctuations:
    stops.append(p)
stops.append('')

# arg parser
parser = argparse.ArgumentParser(description='Builds the models to test accuracy')
parser.add_argument('-l', '--lab_d', default='reference_data.csv')
parser.add_argument('-d', '--dirName', default=None)

args = parser.parse_args()
dirName = args.dirName

if not dirName:
    current_directory = os.getcwd()
    dirName = f'{current_directory}/Data/'

dirHeirarchy = f'{dirName}Brands/'
Saves = f'{dirName}SavedModels/'
lab_d = args.lab_d


def getVocab(comment):
    lemmatized = []
    tokens = nlp(unidecode.unidecode(str(comment).lower()))
    for token in tokens:
        token = token.lemma_
        if token not in stops:
            lemmatized.append(token)
    return lemmatized


if __name__ == "__main__":

    lab_df = pd.read_csv(lab_d)

    for brand in os.listdir(dirHeirarchy):
        brand_path = os.path.join(dirHeirarchy, brand)
        if not os.path.isdir(brand_path):
            continue

        allVocab = []
        for root, dirs, files in os.walk(brand_path):
            for file in files:
                if file.endswith(".csv"):
                    fileName = os.path.join(root, file)
                    # print('Getting vocab from' , fileName)

                    csv = pd.read_csv(fileName)
                    csv['query_vocab'] = csv['query'].apply(getVocab)
                    query_vocab_list = csv['query_vocab'].sum()

                    csv['brand_vocab'] = csv['brand'].apply(getVocab)
                    brand_vocab_list = csv['brand_vocab'].sum()

                    csv['product_vocab'] = csv['product_name'].apply(getVocab)
                    product_vocab_list = csv['product_vocab'].sum()

                    allVocab = list(set(query_vocab_list + brand_vocab_list + product_vocab_list + allVocab))

        # saving all vocab for the specific brand
        with open('{}{}_vocab.txt'.format(Saves, brand), 'wb') as fp:
            pickle.dump(allVocab, fp)
