# Automatic query 'relevance' to advertisement detection tool based on deep learning, machine learning and natural language processing 

Companies pay search engines like www.google.com to provide advertisements of their brands and products. Google charges companies for each advertisement it shows for a specific user query. However, not all advertisements provided by google relevant to the user-query. While google refunds the add cost for irrelevant queries, companies face the challenge of manually qualifying whenever google shows a wrong add for a brand or a product.
This deep learning, machine learning and NLP -based project solves this issue by automatically classifying irrelevant queries from the relevant ones. The end goal of this project is to help companies in better CPR (Conversion Rate Prediction), thereby saving money on advertisements of their products. The end-to-end AI pipeline performs all the necessary steps as follows:

## 0. Preprocessing:
[https://github.com/Aaditya-Bhatia/Search_Engine_Query-Advertisment_Management/blob/master/0_validate_labeled_data.py] This is the preprocessing script where the labeled data can be checked for manual errors, and cleaned to make the data consistent.

## 1. File Tree Structure:
[https://github.com/Aaditya-Bhatia/Search_Engine_Query-Advertisment_Management/blob/master/1_build_files.py] 
Build a directory hierarchy in terms of a tree structure. The labeled data consists of multiple brands and each brand consists of multiple products.


## 2. Vocabulary Word Embeddings:
[https://github.com/Aaditya-Bhatia/Search_Engine_Query-Advertisment_Management/blob/master/2_build_vocab.py] 
This module uses NLP to convert text used for each brand and each product into "Count Vectorizer" word embeddings. The vocabulary files are saved in the result directories. 

## 3.a. Build Aggregate Model:
[https://github.com/Aaditya-Bhatia/Search_Engine_Query-Advertisment_Management/blob/master/3a_build_aggregate_model.py] 
Not all brands or products have enough datapoints to build Machine Learning models. To categorize the queries for the brands and products that have  inadequate model features, the aggregate model which is trained on all the labeled data points. Since "all" datapoints encompass a large number of training datapoints, a keras-based deep learning classifier is used as opposed to a machine learner classifier which usually provides lower model-performance-metrics in such NLP-based use cases. The vocabulary features created in step 2 are used to train deep learning models to predict a user query relates to "some or the other" brand or product available with the company. This aggregate model is trained on "all" labeled dataset, and does not determine if a "specific" brand or "specific" product information is provided. 

## 3.b. Build Specific Models:
[https://github.com/Aaditya-Bhatia/Search_Engine_Query-Advertisment_Management/blob/master/3b_build_aggregate_model.py]
If adequate data points are available for building a machine learning classifier then this step is automatically performed under the pipeline. Random Forest is known to be a robust ML classifier for NLP operations. For each brand and product in the directory structure, their specific model is created. Automatic qualification of an "acceptable" model is performed and the models which lower AUCs (if the model is performing random guesses) then it is discarded. Through Hyperparameter tuning the models are optimized for best AUC and Accuracy.

## 3.b. Retraining models for later use:
[https://github.com/Aaditya-Bhatia/Search_Engine_Query-Advertisment_Management/blob/master/3c_retrain.py]
Companies have an influx of new products and removing old products. As time passes, labeled data also changes. This module automatically detects these changes and retrains the specific models where changes have been made.

## 4. Get Predictions on new data:
This module gets predictions for new data. This can be customized for individual company needs.

A sample image of Canon EOS search query for which many companies paid for providing a "relevant" advertisement.
![ad_img](https://github.com/Aaditya-Bhatia/Search_Engine_Query-Advertisment_Management/blob/master/Canon_Add_Image.png)

