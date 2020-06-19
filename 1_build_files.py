#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 10:52:20 2020

@author: aadityabhatia

Info: This module builds a hierarchical structure for brands and products where 
Brands/ foler has multiple brands and 
each brand has multiple product folders.
Brand folders have brand csvs and product folders have product csvs.
"""

import pandas as pd
import os
import spacy
import unidecode
from nltk.corpus import stopwords
import string
import argparse

nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
stops = stopwords.words("english")
punctuations = string.punctuation
for p in punctuations:
    stops.append(p)
stops.append('')


# arg parser
parser = argparse.ArgumentParser(description='Builds the models to test accuracy')
parser.add_argument('-d', '--dirName', default=None)
parser.add_argument('-l', '--lab_d', default='reference_data.csv')

args = parser.parse_args()
dirName = args.dirName
if not dirName:
    current_directory = os.getcwd()
    current_directory = current_directory.replace('\\', '/')
    dirName = f'{current_directory}/Data/'

lab_d = args.lab_d


def getLemma(comment):
    lemmatized = []
    tokens = nlp(unidecode.unidecode(str(comment).lower()))
    for token in tokens:
        token = token.lemma_
        if token not in stops:
            lemmatized.append(token)
    return ' '.join(lemmatized)


if __name__ == "__main__":
    lab_df = pd.read_csv(lab_d)

    # Preprocessing the labels file for future use:
    lab_df['query_lemma'] = lab_df['query'].apply(getLemma)

    if not os.path.exists(dirName):
        os.mkdir(dirName)

    # making brand path
    if not os.path.exists(dirName + 'Brands'):
        os.mkdir(dirName + 'Brands')
    if not os.path.exists(dirName + 'SavedModels'):
        os.mkdir(dirName + 'SavedModels')

    # lists for unprocessed brands (can save for if needed)
    unprocessed_brands = []
    unprocessed_products = []

    # making file structures in local dir
    for brand, brand_grouper in lab_df.groupby('brand'):
        brand = brand.replace('/', '-')
        brand_path = dirName + 'Brands/' + brand + '/'

        # making brand files
        if len(brand_grouper) > 5:
            if not os.path.exists(brand_path):
                os.mkdir(brand_path)
            brand_grouper.to_csv(brand_path + brand + '.csv')

            # making product files
            for product, product_grouper in brand_grouper.groupby('product_id'):
                product_path = brand_path + product
                if len(product_grouper) > 5:
                    if not os.path.exists(product_path):
                        os.mkdir(product_path)
                    product_grouper.to_csv(product_path + '/' + product + '.csv')
                else:
                    unprocessed_products.append(product)
        else:
            unprocessed_brands.append(brand)

    print('\n\n\n...Estimations')
    print(len(unprocessed_brands) / lab_df.brand_id.nunique() * 100, '% brands not processed')
    print(len(unprocessed_products) / lab_df.product_id.nunique() * 100, '% products not processed')
