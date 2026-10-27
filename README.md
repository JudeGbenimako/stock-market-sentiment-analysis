# Stock Market Sentiment Analysis Using Tweets

## Overview

This academic project explores **financial tweet sentiment classification** using Natural Language Processing (NLP) and machine learning. The objective is to classify each tweet as **Bearish (0), Bullish (1), or Neutral (2)**. It does **not** predict stock prices or verify market movements.

Developed for the **Text Mining** course in the MSc in Data Science and Advanced Analytics at **NOVA Information Management School (NOVA IMS)**, 2024/2025.

## Data

- Training data: **9,543 labeled financial tweets**
- Test data: **2,388 unlabeled tweets**
- Classes: Bearish (0), Bullish (1), and Neutral (2)
- Class distribution reported in the original analysis: approximately 15.1% bearish, 20.2% bullish, and 64.7% neutral

The datasets (`train.csv` and `test.csv`) are **not included** in the repository.

## Methods

- **Exploration and preprocessing:** tweet cleaning, word counts, tokenization, lemmatization, and text normalization
- **Representations:** Bag-of-Words (BoW), TF-IDF, Word2Vec, and Sentence-BERT (SBERT)
- **Model comparison:** Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost, voting ensembles, and exploratory pre-trained transformer approaches (BERT, RoBERTa, FinBERT)
- **Evaluation:** train/validation splits, accuracy, precision, recall, and macro F1-score
- **Class imbalance:** experiments included oversampling/SMOTE; the final submission pipeline used random oversampling on its training partition

## Recorded results

| Evaluation | Model / representation | Accuracy | Macro F1 |
|---|---|---:|---:|
| Main model comparison | Logistic Regression + Bag-of-Words (tweet-tailored preprocessing) | 0.8072 | 0.7199 |
| Final submission pipeline | Logistic Regression + CountVectorizer / TF-IDF + random oversampling | 0.80 | 0.73 |

These values come from **different recorded validation runs**, including different splits and preprocessing/modeling procedures; they should not be interpreted as a controlled head-to-head comparison. Metrics are reported from the original notebooks and have **not been recomputed** for the cleaned copies.

The original model comparison identified Logistic Regression with Bag-of-Words as one of the best-performing experimental approaches. The final submission used a CountVectorizer + TF-IDF Transformer pipeline with Logistic Regression.

## Repository contents

- [stock_sentiment_analysis.ipynb](stock_sentiment_analysis.ipynb) — Exploratory data analysis, tweet distributions, text cleaning, and visualizations
- [sentiment_model_experiments.ipynb](sentiment_model_experiments.ipynb) — Comparisons of text representations, preprocessing methods, and classification models
- [sentiment_additional_experiments.ipynb](sentiment_additional_experiments.ipynb) — Further experiments with preprocessing and classification pipelines
- [sentiment_final_pipeline.ipynb](sentiment_final_pipeline.ipynb) — Final classification pipeline, validation, and prediction-generation workflow
- [requirements.txt](requirements.txt) — Python package dependencies for the notebook workflows
- [.gitignore](.gitignore) — Excludes local datasets, generated predictions, and notebook caches

## Reproducibility and limitations

The notebooks require the original `train.csv` and `test.csv` and may require downloading NLP resources or pretrained models. They document the **2025 academic workflows** and have not been tested against current package versions. The model experiments, additional experiments, and final pipeline notebooks have been cleared of saved execution outputs. The EDA notebook retains saved analysis output and visualizations. Student identification and contact details were removed from the publication copies, but older Git commits may still contain previously published information.

The final notebook performs inference using the original test text rather than the cleaned test-text column; this should be reviewed before applying the workflow to new datasets. The primary objective is **tweet sentiment classification**, not stock-market return prediction.

## Tools

Python, Jupyter Notebook, Pandas, NumPy, Scikit-learn, NLTK, Gensim, XGBoost, imbalanced-learn, Transformers, Sentence-Transformers, Matplotlib, and Seaborn.

## Team

**Group 23, NOVA IMS:** Ana Caleiro, André Lourenço, Duarte Marques, Érica Parracho, and Jude Gbenimako.
