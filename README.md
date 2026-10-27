# Stock Market Sentiment Analysis Using Tweets

## Project Overview

This project explores the use of Natural Language Processing (NLP) and machine learning to classify stock market sentiment from tweets.

The objective is to identify whether a tweet expresses **Bearish, Bullish, or Neutral** sentiment. Different text preprocessing methods, feature extraction techniques, and classification models were evaluated to identify an effective approach.

This project was developed as part of the **Text Mining course** in the MSc in Data Science and Advanced Analytics program at NOVA Information Management School (NOVA IMS).

## Dataset

The project uses two datasets containing financial tweets:

- **Training dataset:** 9,543 labeled tweets
- **Test dataset:** 2,388 unlabeled tweets

Tweets are classified into three sentiment categories:

| Label | Sentiment | Distribution |
|---|---|---|
| 0 | Bearish | 15.1% |
| 1 | Bullish | 20.2% |
| 2 | Neutral | 64.7% |

The dataset is imbalanced, with Neutral tweets representing the majority of observations.

## Methodology

The analysis followed several stages:

1. **Exploratory Data Analysis:** Examined sentiment distributions, word frequencies, and text characteristics.
2. **Text Preprocessing:** Applied different cleaning strategies to remove noise, URLs, mentions, punctuation, and other unnecessary characters.
3. **Feature Engineering:** Explored Bag-of-Words (BoW), TF-IDF, Word2Vec, and Sentence-BERT (SBERT).
4. **Model Development:** Evaluated traditional machine learning classifiers, ensemble methods, and transformer-based models.
5. **Model Evaluation:** Compared model performance using accuracy, precision, recall, and macro F1-score.

The final pipeline used tweet-specific preprocessing, CountVectorizer, TfidfTransformer, and Logistic Regression with a stratified training and validation split.

## Models Evaluated

- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes
- Random Forest
- XGBoost
- Voting Classifier
- BERT
- RoBERTa
- FinBERT

## Results

The following were among the best-performing configurations in the model comparison:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression + BoW | 0.81 | 0.72 |
| SVM + TF-IDF | 0.80 | 0.71 |
| SVM + BoW | 0.79 | 0.71 |
| Optimized SVM + TF-IDF (SMOTE) | 0.78 | 0.70 |
| Random Forest + BoW | 0.80 | 0.69 |

**Logistic Regression with Bag-of-Words achieved the highest reported macro F1-score of 0.72 and accuracy of 81% in the model comparison.**

Macro F1-score was prioritized because of the class imbalance, ensuring that performance across all three sentiment categories was considered.

## Key Findings

- Traditional machine learning models with sparse text representations outperformed the tested transformer-based models.
- Tweet-specific preprocessing improved classification performance.
- Logistic Regression and SVM provided strong results.
- Class imbalance significantly influenced model evaluation and selection.
- More complex models did not necessarily produce better results on this dataset.

The project evaluates sentiment classification rather than directly predicting actual stock price movements.

## Technologies Used

Python, Jupyter Notebook, Pandas, NumPy, Scikit-learn, NLTK, Gensim, XGBoost, Transformers, Sentence-Transformers, Matplotlib, and Seaborn.

## Project Files

- [stock_sentiment_analysis.ipynb](stock_sentiment_analysis.ipynb) — Jupyter Notebook containing the analysis and modeling experiments.
- `README.md` — Project documentation.

The original training and test datasets are not included in this repository.

## Project Team

Developed as a group project (Group 23), NOVA IMS, 2025.

- Ana Caleiro
- André Lourenço
- Duarte Marques
- Érica Parracho
- Jude Gbenimako

## Academic Context

**Institution:** NOVA Information Management School (NOVA IMS)

**Program:** MSc in Data Science and Advanced Analytics

**Course:** Text Mining

**Year:** 2025
