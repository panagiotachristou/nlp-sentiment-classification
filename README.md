# Sentiment classification of movie reviews (NLP – Exercise 9)

Binary sentiment classification (positive / negative) of movie reviews with logistic regression,
compared to a majority baseline. Two kinds of features are compared:
TF-IDF (compressed with truncated SVD) and centroids of pre-trained GloVe word embeddings.
Based on the course lab notebooks on text classification.

## Contents
- `2nd_nlp_text_classification.ipynb` – all code, with outputs
- `data/review_polarity.tar.gz` – the dataset (Cornell movie review polarity dataset v2.0, 2,000 reviews)
- `report.pdf` – the report
- `requirements.txt` – Python packages

## How to run
1. Open the notebook in Google Colab (link at the top of the notebook) or in Jupyter.
2. Run the install cells, then restart the session (Colab: Runtime → Restart session).
3. Run all cells.

The dataset is included in `data/`. If it is missing, the notebook downloads it automatically
and checks its SHA-256 checksum. The GloVe embeddings (`glove-wiki-gigaword-100`, ~130 MB)
are downloaded automatically with gensim.

To run locally: `pip install -r requirements.txt`

## Method
- Train / development / test split: 1,050 / 350 / 600 reviews (stratified)
- Text normalization (lowercase, negations expanded)
- Features:
  - TF-IDF unigrams + bigrams (5,000 features), compressed with truncated SVD (k components)
  - Centroids of GloVe embeddings (100 dimensions): plain average and IDF-weighted average
- Logistic regression with L2 regularization (C)
- All settings tuned on the development set; the test set is used only once

## Results (macro-F1)
| Classifier | Development | Test |
|---|---|---|
| Majority baseline | 0.333 | 0.333 |
| Logistic regression, TF-IDF + SVD (k = 500, C = 100) | 0.874 | 0.850 |
| Logistic regression, GloVe centroids (C = 10) | 0.771 | [ ] |
| Logistic regression, GloVe IDF-weighted centroids (C = 100) | 0.737 | [ ] |

Author: Panagiota Christou
