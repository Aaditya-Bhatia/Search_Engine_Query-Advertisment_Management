# Automatic Query Relevance Detection for Advertisement

Improve your advertisement efficiency and conversion rate prediction (CPR) with this AI-based tool that leverages deep learning, machine learning, and natural language processing (NLP) to automatically classify irrelevant queries from relevant ones. This project is designed to save money on advertisements by identifying when search engines like Google display wrong ads for a brand or product.

## Features

- Validate and preprocess labeled data
- Build a file tree structure for multiple brands and products
- Create vocabulary word embeddings using NLP
- Train an aggregate deep learning model for general query classification
- Train specific machine learning models for each brand and product
- Automatically retrain models as labeled data changes
- Obtain predictions for new data

## Getting Started

### 0. Preprocessing

```bash
python src/0_validate_labeled_data.py
```

Validate and clean the labeled data for consistency.

### 1. File Tree Structure

```bash
python src/1_build_files.py
```

Organize the labeled data into a tree structure consisting of multiple brands and products.

### 2. Vocabulary Word Embeddings

```bash
python src/2_build_vocab.py
```

Convert text for each brand and product into "Count Vectorizer" word embeddings using NLP. Save vocabulary files in the result directories.

### 3.a. Build Aggregate Model

```bash
python src/3a_build_aggregate_model.py
```

Train an aggregate deep learning model on all labeled data points using Keras. This model classifies whether a query relates to "some or the other" brand or product, but not specific ones.

### 3.b. Build Specific Models

```bash
python src/3b_build_aggregate_model.py
```

Create specific Random Forest models for each brand and product in the directory structure, discarding those with low AUCs. Optimize the models for best AUC and accuracy using hyperparameter tuning.

### 3.c. Retraining Models for Later Use

```bash
python src/3c_retrain.py
```

Automatically detect changes in labeled data and retrain the specific models accordingly.

### 4. Get Predictions on New Data

Customize this module to obtain predictions for new data based on individual company needs.

## Sample Image

An example of a Canon EOS search query for which many companies paid to display a "relevant" advertisement:

![ad_img](https://github.com/Aaditya-Bhatia/Search_Engine_Query-Advertisment_Management/blob/master/imgs/Canon_Add_Image.png)
